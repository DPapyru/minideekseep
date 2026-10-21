import copy
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import TokenStream, fingerprint, verify_corpus
from .model import MiniDeepSeek


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, lr=3e-4):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or "norm" in name else decay).append(parameter)
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)],
        lr=lr,
        betas=(0.9, 0.95),
        foreach=False,
    )


@contextmanager
def routing_counts(model):
    handles = []
    for block in model.layers:
        moe = block.moe

        def record(module, inputs, output, moe=moe):
            if not torch.is_grad_enabled():
                return
            # Checkpoint recomputation is disabled for counters by removing hooks before backward.
            with torch.no_grad():
                scores = torch.nn.functional.softplus(output.float()).sqrt()
                idx = (scores + moe.route_bias).topk(moe.top_k, -1).indices
                moe.route_counts.add_(torch.bincount(idx.flatten(), minlength=len(moe.experts)))

        handles.append(moe.router.register_forward_hook(record))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def save_checkpoint(path, model, optimizer, stream, step, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(
        format_version=1,
        model=model.state_dict(),
        config=model.config.to_dict(),
        optimizer=optimizer.state_dict(),
        stream=stream.state_dict() if stream else None,
        step=step,
        metadata=metadata or {},
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_latest(directory, model, optimizer, stream, step, metadata):
    directory = Path(directory)
    pending = directory / "next.pt"
    save_checkpoint(pending, model, optimizer, stream, step, metadata)
    if (directory / "latest.pt").exists():
        (directory / "latest.pt").replace(directory / "previous.pt")
    pending.replace(directory / "latest.pt")


def load_checkpoint(path, device="cpu", restore_rng=False):
    # Only load checkpoints produced locally or from a trusted source (optimizer/RNG use pickle).
    state = torch.load(path, map_location="cpu", weights_only=False)
    values = state["config"].copy()
    for name in ("engram_layers", "engram_buckets"):
        values[name] = tuple(values[name])
    model = MiniDeepSeek(ModelConfig(**values)).to(device)
    model.load_state_dict(state["model"])
    model.set_fake_quantization(state["metadata"].get("fake_quant", False))
    if restore_rng:
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return model, state


def evaluate(model, stream, batches=10, device="cpu"):
    was_training = model.training
    model.eval()
    losses, tokens = 0.0, 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = stream.next(device)
            output = model.forward_train(x, y)
            n = int((y != -100).sum())
            losses += output.lm_loss.item() * n
            tokens += n
    model.train(was_training)
    loss = losses / max(tokens, 1)
    return dict(lm_loss=loss, perplexity=math.exp(min(loss, 50)), tokens=tokens)


def compare_prefill(model, stream, batches=10, device="cpu"):
    """Measure approximate replay against full prefill on identical valid prompts."""
    model.eval()
    rows = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = stream.next(device)
            valid = int((y != -100).sum())
            x, target = x[:, :valid], y[:, valid - 1]
            row = {}
            for bounded, key in ((False, "exact"), (True, "bounded")):
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.monotonic()
                logits, _ = model.prefill(x, bounded=bounded)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                row[key + "_seconds"] = time.monotonic() - start
                row[key + "_loss"] = torch.nn.functional.cross_entropy(logits, target).item()
            rows.append(row)
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def train(
    data_dir,
    run_dir,
    config,
    token_budget=1_000_000,
    device="cuda",
    length=256,
    tokens_per_step=4096,
    resume=None,
    max_steps=None,
    lr=3e-4,
    initialize=None,
    replay=False,
    fake_quant=False,
):
    seed_everything(42)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and initialize:
        raise ValueError("Choose resume or initialization, not both")
    if length > config.max_seq_len or min(length, tokens_per_step, token_budget) < 1:
        raise ValueError("Invalid sequence length or training budget")
    if (run_dir / "latest.pt").exists() and not resume:
        raise FileExistsError("Run already has a checkpoint; resume it or choose another directory")
    manifest = verify_corpus(data_dir)
    metadata = dict(
        tokenizer_sha256=fingerprint(Path(data_dir) / "tokenizer.json"),
        data_manifest_sha256=fingerprint(Path(data_dir) / "encoded.json"),
        token_budget=token_budget,
        tokens_per_step=tokens_per_step,
        lr=lr,
        replay=replay,
        fake_quant=fake_quant,
    )
    if config.vocab_size < manifest["vocab_size"]:
        raise ValueError("Model vocabulary is smaller than the trained tokenizer")
    stream = TokenStream(data_dir, length=length)
    validation = TokenStream(data_dir, "validation", length, seed=123)
    start_step = 0
    if resume:
        model, saved = load_checkpoint(resume, device, restore_rng=True)
        if model.config != config:
            raise ValueError("Resume model config differs from the requested config")
        for key in metadata:
            if saved["metadata"].get(key) != metadata[key]:
                raise ValueError(f"Resume metadata mismatch: {key}")
        stream.load_state_dict(saved["stream"])
        start_step = saved["step"]
    else:
        if initialize:
            model, initialized = load_checkpoint(initialize, device)
            if initialized["metadata"].get("tokenizer_sha256") != metadata["tokenizer_sha256"]:
                raise ValueError("Initialization tokenizer mismatch")
            if model.config != config:
                raise ValueError("Initialization config mismatch")
        else:
            model = MiniDeepSeek(config).to(device)
    model.set_fake_quantization(fake_quant)
    optimizer = make_optimizer(model, lr)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
    best = saved["metadata"].get("best_loss", float("inf")) if resume else float("inf")
    if best < float("inf"):
        metadata["best_loss"] = best
    last_save = time.monotonic()
    step = start_step
    started = time.monotonic()
    initial_tokens = stream.tokens
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        while stream.tokens < token_budget and (max_steps is None or step - start_step < max_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = stream.tokens / token_budget
            scale = (
                progress / 0.02
                if progress < 0.02
                else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (progress - 0.02) / 0.98)))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr * max(scale, 1 / 1000)
            # Weight every microbatch by its real targets, not by its padded length.
            batches = []
            count = 0
            while count < tokens_per_step:
                batch = stream.next(device)
                batches.append(batch)
                count += int((batch[1] != -100).sum())
            lm_sum = aux_sum = 0.0
            for x, y in batches:
                weight = int((y != -100).sum()) / count
                with routing_counts(model):
                    # Replay loss uses a valid unpadded prefix for each single-document sample.
                    n = int((y != -100).sum())
                    output = (
                        model.forward_replay(x[:, :n], y[:, :n])
                        if replay
                        else model.forward_train(x[:, :n], y[:, :n])
                    )
                (output.loss * weight).backward()
                lm_sum += output.lm_loss.item() * weight
                aux_sum += output.index_loss.item() * weight
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            route_fractions = [
                (layer.moe.route_counts / layer.moe.route_counts.sum().clamp_min(1)).tolist()
                for layer in model.layers
            ]
            model.update_router_biases()
            step += 1
            row = dict(
                step=step,
                tokens=stream.tokens,
                lm_loss=lm_sum,
                index_loss=aux_sum,
                grad_norm=float(norm),
                lr=optimizer.param_groups[0]["lr"],
                route_fractions=route_fractions,
                tokens_per_second=(stream.tokens - initial_tokens) / (time.monotonic() - started),
            )
            if str(device).startswith("cuda"):
                row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.max_memory_reserved() > 2.6 * 2**30:
                    raise RuntimeError(
                        "Memory budget exceeded; retry length=128, keep model dimensions"
                    )
            if step % 100 == 0:
                val_state = copy.deepcopy(validation.state_dict())
                row["validation"] = evaluate(model, validation, device=device)
                validation.load_state_dict(val_state)
                if row["validation"]["lm_loss"] < best:
                    best = row["validation"]["lm_loss"]
                    metadata["best_loss"] = best
                    save_checkpoint(run_dir / "best.pt", model, optimizer, stream, step, metadata)
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if time.monotonic() - last_save > 900:
                save_latest(run_dir, model, optimizer, stream, step, metadata)
                last_save = time.monotonic()
    except KeyboardInterrupt:
        print(
            "Interrupted: resume from the last completed checkpoint; partial step discarded",
            flush=True,
        )
        raise
    finally:
        # On interruption, only completed optimizer steps are reproducible. Do not save partial state.
        optimizer.zero_grad(set_to_none=True)
    save_latest(run_dir, model, optimizer, stream, step, metadata)
    return model


def benchmark(config, length=128, device="cuda", steps=100, seconds=600, tokens_per_step=4096):
    seed_everything(42)
    model = MiniDeepSeek(config).to(device).train()
    optimizer = make_optimizer(model)
    x = torch.randint(5, config.vocab_size, (1, length + 1), device=device)
    if str(device).startswith("cuda"):
        if torch.cuda.get_device_capability() != (7, 5):
            print(
                "GPU differs from GTX 1650; reporting measured results for this device", flush=True
            )
        torch.cuda.reset_peak_memory_stats()
    accumulation = max(1, math.ceil(tokens_per_step / length))
    start = time.monotonic()
    completed = 0
    while completed < steps or time.monotonic() - start < seconds:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with routing_counts(model):
                output = model.forward_train(x[:, :-1], x[:, 1:])
            (output.loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.update_router_biases()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            if torch.cuda.max_memory_reserved() > 2.6 * 2**30:
                raise RuntimeError("Peak reserved memory exceeds 2.6 GiB; retry shorter sequences")
        completed += 1
        print(
            json.dumps(
                dict(step=completed, loss=output.lm_loss.item(), seconds=time.monotonic() - start)
            ),
            flush=True,
        )
    elapsed = time.monotonic() - start
    speed = completed * length * accumulation / elapsed
    actual_tokens_per_step = length * accumulation
    budget = int(min(50_000_000, speed * 72 * 3600)) // actual_tokens_per_step * actual_tokens_per_step
    return dict(
        config=config.to_dict(),
        sequence_length=length,
        tokens_per_step=actual_tokens_per_step,
        steps=completed,
        seconds=elapsed,
        tokens_per_second=speed,
        estimated_72h_tokens=budget,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20
        if str(device).startswith("cuda")
        else 0,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20
        if str(device).startswith("cuda")
        else 0,
    )
