"""A small semi-autoregressive drafter and exact rejection verifier.

The verifier restores its immutable cache at the first rejection. Verification is
sequential in this reference implementation; throughput must be measured before use.
"""

import json
import time
import weakref
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .attention import LocalMemory
from .data import TokenStream, fingerprint
from .layers import HyperConnection, RMSNorm, rope
from .model import Block
from .training import load_checkpoint, make_optimizer, routing_counts, seed_everything


class DSpark(nn.Module):
    def __init__(self, backbone, width=256, markov_rank=32, positions=3):
        super().__init__()
        self._backbone = weakref.ref(backbone)
        self.positions = positions
        c = replace(backbone.config, expert_dim=width, engram_layers=())
        self.config = c
        self.condition = nn.Linear(c.dim, c.dim, bias=False)
        self.position = nn.Parameter(torch.randn(positions, c.dim) * 0.02)
        self.layers = nn.ModuleList([Block(c, 0), Block(c, 0)])
        self.norm = RMSNorm(c.dim)
        self.markov_embed = nn.Embedding(c.vocab_size, markov_rank)
        self.markov_head = nn.Linear(markov_rank, c.vocab_size, bias=False)
        self.confidence = nn.Linear(c.dim + markov_rank, 1)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    @property
    def backbone(self):
        model = self._backbone()
        if model is None:
            raise RuntimeError("Keep the backbone alive while using its shared embedding and head")
        return model

    def base(self, feature, last_token):
        c = self.config
        b = last_token.size(0)
        if feature.ndim == 2:
            feature = feature[:, None]
        context = self.condition(feature[:, -c.window :])
        context_length = context.size(1)
        ids = last_token.new_full((b, self.positions), c.noise_id)
        ids[:, 0] = last_token
        h = self.backbone.embedding(ids) + self.position
        h = h.unsqueeze(2).expand(-1, -1, c.hc_streams, -1)
        pre = h.new_zeros(b, self.positions, c.hc_streams)
        pre[..., 0] = 1
        for block in self.layers:
            latent = block.attention.kv_norm(block.attention.local_kv(context))
            latent = rope(latent, torch.arange(context_length, device=latent.device), c.rope_dim)
            local = LocalMemory(latent, 0)
            h, pre, *_ = block(
                h, pre, ids, context_length, local, None, None, None, True, None, False
            )
        return self.norm(HyperConnection.collapse(h, pre))

    def conditional(self, hidden, previous):
        markov = self.markov_embed(previous)
        logits = self.backbone.head(hidden) + self.markov_head(markov)
        confidence = self.confidence(torch.cat((hidden, markov), -1)).squeeze(-1)
        return logits, confidence

    @torch.no_grad()
    def propose(self, feature, last_token, temperature=1.0):
        hidden = self.base(feature, last_token)
        probabilities, tokens, confidence = [], [], []
        previous = last_token
        for i in range(self.positions):
            logits, predicted = self.conditional(hidden[:, i], previous)
            q = (logits / temperature).softmax(-1)
            previous = torch.multinomial(q, 1).squeeze(-1)
            tokens.append(previous)
            probabilities.append(q)
            confidence.append(predicted.sigmoid())
        return torch.stack(tokens, 1), probabilities, torch.stack(confidence, 1)


def corrected_distribution(p, q):
    residual = (p - q).clamp_min(0)
    total = residual.sum(-1, keepdim=True)
    # A rejection is impossible when p == q; use p to handle numerical cancellation.
    return torch.where(total > 1e-8, residual / total.clamp_min(1e-8), p)


@torch.no_grad()
def choose_length(confidence, profile):
    survival = confidence.cumprod(-1)[0]

    def score(row):
        length = row["verification_length"]
        expected_tokens = 1 + survival[: length - 1].sum().item()
        return expected_tokens / (row["seconds"] / profile["rounds"])

    return max(profile["curves"], key=score)["verification_length"]


@torch.no_grad()
def speculative_round(
    backbone, draft, prefix, temperature=1.0, verification_length=3, profile=None
):
    if prefix.size(0) != 1 or temperature <= 0:
        raise ValueError("The reference verifier supports batch size one and temperature > 0")
    backbone.eval()
    draft.eval()
    hidden, _, initial_cache = backbone._run(prefix)
    logits = backbone.head(hidden[:, -1])
    tokens, proposals, confidence = draft.propose(hidden, prefix[:, -1], temperature)
    if profile:
        verification_length = choose_length(confidence, profile)
    if prefix.size(1) + verification_length > backbone.config.max_seq_len:
        raise ValueError("Speculative round exceeds the context limit")
    cache = initial_cache
    output = []
    accepted = 0
    for i in range(min(verification_length, draft.positions)):
        p, q = (logits / temperature).softmax(-1), proposals[i]
        token = tokens[:, i]
        ratio = (p.gather(-1, token[:, None]) / q.gather(-1, token[:, None]).clamp_min(1e-9)).clamp(
            max=1
        )
        saved_cache = cache
        if torch.rand_like(ratio).item() <= ratio.item():
            output.append(token.item())
            accepted += 1
            logits, cache = backbone.decode_step(token[:, None], cache)
            if token.item() == backbone.config.eos_id:
                break
        else:
            replacement = torch.multinomial(corrected_distribution(p, q), 1)
            # Cache updates are functional: rejected drafts never mutate the saved prefix state.
            cache = saved_cache
            logits, cache = backbone.decode_step(replacement, cache)
            output.append(replacement.item())
            break
    return (
        output,
        cache,
        dict(
            accepted=accepted,
            proposed=min(verification_length, draft.positions),
            confidence=confidence.tolist(),
        ),
    )


def train_draft(
    checkpoint_path,
    data_dir,
    output_dir,
    device="cuda",
    tokens=1_000_000,
    length=128,
    max_steps=None,
    width=256,
    resume=None,
):
    seed_everything(42)
    backbone, state = load_checkpoint(checkpoint_path, device)
    if fingerprint(Path(data_dir) / "tokenizer.json") != state["metadata"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer mismatch")
    backbone.eval()
    draft = DSpark(backbone, width=width).to(device).train()
    optimizer = make_optimizer(draft)
    stream = TokenStream(data_dir, length=length)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "draft.pt").exists() and not resume:
        raise FileExistsError("Draft run already exists; resume it or choose another directory")
    backbone_digest = fingerprint(checkpoint_path)
    tokenizer_digest = fingerprint(Path(data_dir) / "tokenizer.json")
    from .data import verify_corpus

    verify_corpus(data_dir)
    data_digest = fingerprint(Path(data_dir) / "encoded.json")
    step = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["backbone_sha256"] != backbone_digest or saved["width"] != width:
            raise ValueError("Draft resume requires the same backbone and width")
        if saved["tokenizer_sha256"] != tokenizer_digest or saved["token_budget"] != tokens:
            raise ValueError("Draft resume tokenizer or token budget mismatch")
        if saved["data_manifest_sha256"] != data_digest:
            raise ValueError("Draft resume encoded corpus mismatch")
        draft.load_state_dict(saved["draft"])
        optimizer.load_state_dict(saved["optimizer"])
        stream.load_state_dict(saved["stream"])
        step = saved["step"]
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    initial_step = step
    last_save = time.monotonic()
    target = directory / "draft.pt"

    def save():
        temporary = target.with_suffix(".tmp")
        torch.save(
            dict(
                draft=draft.state_dict(),
                backbone=str(Path(checkpoint_path).resolve()),
                backbone_sha256=backbone_digest,
                tokenizer_sha256=tokenizer_digest,
                data_manifest_sha256=data_digest,
                width=width,
                markov_rank=32,
                positions=3,
                token_budget=tokens,
                step=step,
                stream=stream.state_dict(),
                optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            ),
            temporary,
        )
        temporary.replace(target)

    while stream.tokens < tokens and (max_steps is None or step - initial_step < max_steps):
        x, y = stream.next(device)
        valid = int((y != -100).sum())
        x = x[:, : min(valid, backbone.config.max_seq_len - draft.positions)]
        with torch.no_grad():
            feature, _, cache = backbone._run(x)
            target_logits = backbone.head(feature[:, -1])
        with routing_counts(draft):
            hidden = draft.base(feature.detach(), x[:, -1])
        previous = x[:, -1]
        loss = hidden.sum() * 0
        optimizer.zero_grad(set_to_none=True)
        for i in range(draft.positions):
            logits, confidence = draft.conditional(hidden[:, i], previous)
            with torch.no_grad():
                p = target_logits.softmax(-1)
                q = logits.softmax(-1)
                sampled = torch.multinomial(q, 1)
                acceptance = (p.gather(-1, sampled) / q.gather(-1, sampled).clamp_min(1e-9)).clamp(
                    max=1
                )
                target_logits, cache = backbone.decode_step(sampled, cache)
            loss = loss - (p * logits.log_softmax(-1)).sum(-1).mean()
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                confidence, acceptance.squeeze(-1)
            )
            previous = sampled.squeeze(-1)
        loss = loss / draft.positions
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for layer in draft.layers:
            layer.moe.update_bias()
        step += 1
        print(
            json.dumps(dict(step=step, tokens=stream.tokens, draft_loss=float(loss.detach()))),
            flush=True,
        )
        if time.monotonic() - last_save > 900:
            save()
            last_save = time.monotonic()
    save()
    return target


def profile_draft(backbone, draft, prefix, rounds=5):
    curves = []
    with torch.no_grad():
        for length in range(1, draft.positions + 1):
            start = time.monotonic()
            outputs = accepted = 0
            for _ in range(rounds):
                tokens, _, stats = speculative_round(
                    backbone, draft, prefix, verification_length=length
                )
                outputs += len(tokens)
                accepted += stats["accepted"]
            if prefix.is_cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            curves.append(
                dict(
                    verification_length=length,
                    tokens_per_second=outputs / elapsed,
                    accepted=accepted,
                    seconds=elapsed,
                )
            )
    if prefix.is_cuda:
        torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        for _ in range(rounds):
            logits, cache = backbone.prefill(prefix)
            for _ in range(draft.positions):
                token = torch.multinomial(logits.softmax(-1), 1)
                logits, cache = backbone.decode_step(token, cache)
    if prefix.is_cuda:
        torch.cuda.synchronize()
    baseline = rounds * draft.positions / (time.monotonic() - start)
    return dict(
        rounds=rounds,
        curves=curves,
        autoregressive_tokens_per_second=baseline,
        measured_speedup=max(row["tokens_per_second"] for row in curves) / baseline,
        recommended_verification_length=max(curves, key=lambda row: row["tokens_per_second"])[
            "verification_length"
        ],
        enabled_by_default=False,
    )
