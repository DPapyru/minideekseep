import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import RMSNorm, rope
from .quantization import fake_fp4, fake_fp8


@dataclass
class GlobalMemory:
    values: torch.Tensor
    keys: torch.Tensor
    ends: torch.Tensor
    pending_values: torch.Tensor
    pending_gates: torch.Tensor


@dataclass
class LocalMemory:
    values: torch.Tensor
    start: int


@dataclass
class Selection:
    indices: torch.Tensor
    valid: torch.Tensor
    candidates: torch.Tensor | None = None


def masked_distribution(logits, valid):
    weights = logits.masked_fill(~valid, -1e9).softmax(-1) * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)


class SparseAttention(nn.Module):
    def __init__(self, c, layer_id):
        super().__init__()
        self.c = c
        self.fake_quant = False
        self.mode = c.modes[layer_id]
        self.decoder = layer_id >= c.encoder_layers
        self.first_decoder = layer_id == c.encoder_layers
        self.ratio = 1 if self.decoder else c.encoder_compression
        self.q_a = nn.Linear(c.dim, c.q_rank, bias=False)
        self.q_norm = RMSNorm(c.q_rank, c.norm_eps)
        self.q_b = nn.Linear(c.q_rank, c.heads * c.head_dim, bias=False)
        self.local_kv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.out_a = nn.Parameter(
            torch.empty(c.output_groups, c.output_rank, c.heads // c.output_groups * c.head_dim)
        )
        self.out_b = nn.Linear(c.output_groups * c.output_rank, c.dim, bias=False)
        self.sink = nn.Parameter(torch.zeros(c.heads))
        if self.mode == "full":
            self.global_kv = nn.Linear(c.dim, c.head_dim, bias=False)
            self.global_norm = RMSNorm(c.head_dim, c.norm_eps)
            self.compress_gate = (
                nn.Linear(c.dim, c.head_dim, bias=False) if self.ratio > 1 else None
            )
            self.index_key = nn.Linear(c.head_dim, c.index_dim, bias=False)
            self.index_norm = RMSNorm(c.index_dim, c.norm_eps)
        if self.mode in ("full", "reindex"):
            self.index_query = nn.Linear(c.q_rank, c.index_heads * c.index_dim, bias=False)
            self.index_weights = nn.Linear(c.dim, c.index_heads, bias=False)
        nn.init.normal_(self.out_a, std=0.02)

    def extend_global(self, source, start, old=None):
        c, ratio = self.c, self.ratio
        projected = self.global_kv(source)
        gates = (
            self.compress_gate(source)
            if self.compress_gate is not None
            else torch.zeros_like(projected)
        )
        if old is not None:
            projected = torch.cat((old.pending_values, projected), 1)
            gates = torch.cat((old.pending_gates, gates), 1)
            source_start = start - old.pending_values.size(1)
        else:
            source_start = start
        complete = projected.size(1) // ratio * ratio
        grouped = projected[:, :complete].reshape(
            source.size(0), complete // ratio, ratio, c.head_dim
        )
        weights = (
            gates[:, :complete]
            .reshape(source.size(0), complete // ratio, ratio, c.head_dim)
            .softmax(2)
        )
        latent = self.global_norm((grouped * weights).sum(2))
        positions = torch.arange(source_start, source_start + complete, ratio, device=source.device)
        key = self.index_norm(self.index_key(latent))
        key = rope(key, positions, c.rope_dim, theta=40000.0)
        values = rope(latent, positions, c.rope_dim, theta=40000.0)
        if self.fake_quant:
            values, key = fake_fp4(values), fake_fp4(key, 32)
        ends = positions + ratio - 1
        if old is not None:
            values = torch.cat((old.values, values), 1)
            key = torch.cat((old.keys, key), 1)
            ends = torch.cat((old.ends, ends))
        return GlobalMemory(values, key, ends, projected[:, complete:], gates[:, complete:])

    def select(self, x, qr, q, positions, memory, previous, with_loss):
        c = self.c
        b, s, _ = x.shape
        g = memory.values.size(1)
        zero = x.sum() * 0
        if self.mode == "reuse":
            if previous is None:
                raise ValueError("Reuse without a preceding index source")
            return previous, zero
        if not g:
            return Selection(
                torch.empty(b, s, 0, dtype=torch.long, device=x.device),
                torch.empty(b, s, 0, dtype=torch.bool, device=x.device),
            ), zero
        iq = self.index_query(qr).view(b, s, c.index_heads, c.index_dim)
        iq = rope(iq, positions, c.rope_dim, theta=40000.0)
        if self.fake_quant:
            iq = fake_fp4(iq, 32)
        head_weights = self.index_weights(x) / math.sqrt(c.index_heads * c.index_dim)
        selections, validities, pools, losses = [], [], [], []
        for lo in range(0, s, c.query_chunk):
            hi = min(s, lo + c.query_chunk)
            visible = (memory.ends[None, None, :] <= positions[None, lo:hi, None]).expand(b, -1, -1)
            candidate_ids = None
            if self.mode == "reindex" and self.decoder and previous.candidates is not None:
                pool = previous.candidates[:, lo:hi] & visible
                limit = min(g, c.candidate_blocks * c.candidate_block_size)
                candidate_ids = pool.int().argsort(dim=-1, descending=True, stable=True)[
                    ..., :limit
                ]
                visible = pool.gather(-1, candidate_ids)
                batch = torch.arange(b, device=x.device)[:, None, None]
                keys = memory.keys[batch, candidate_ids]
                scores = torch.einsum("bqhd,bqkd->bqhk", iq[:, lo:hi], keys).relu()
            else:
                scores = torch.einsum("bqhd,bgd->bqhg", iq[:, lo:hi], memory.keys).relu()
            scores = (scores * head_weights[:, lo:hi, :, None]).sum(2)
            masked = scores.masked_fill(~visible, -torch.inf)
            k = min(c.index_top_k, scores.size(-1))
            idx = masked.argsort(dim=-1, descending=True, stable=True)[..., :k]
            values = masked.gather(-1, idx)
            selections.append(candidate_ids.gather(-1, idx) if candidate_ids is not None else idx)
            validities.append(values.isfinite())
            if self.first_decoder:
                # Whole candidate blocks are selected, but never reveal future entries.
                padded = F.pad(masked, (0, (-g) % c.candidate_block_size), value=-torch.inf)
                block_scores = padded.unflatten(-1, (-1, c.candidate_block_size)).amax(-1)
                count = min(c.candidate_blocks, block_scores.size(-1))
                bi = block_scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
                bv = block_scores.gather(-1, bi)
                pool = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(
                    -1, bi, bv.isfinite()
                )
                pools.append(pool.repeat_interleave(c.candidate_block_size, -1)[..., :g] & visible)
            if with_loss:
                # A detached dense global teacher trains index scores beyond the discrete Top-K.
                with torch.no_grad():
                    if candidate_ids is not None:
                        teacher_values = memory.values.detach()[batch, candidate_ids]
                        teacher_logits = torch.einsum(
                            "bqhd,bqkd->bqhk", q[:, lo:hi].detach(), teacher_values
                        )
                    else:
                        teacher_logits = torch.einsum(
                            "bqhd,bgd->bqhg", q[:, lo:hi].detach(), memory.values.detach()
                        )
                    teacher_logits = teacher_logits / math.sqrt(c.head_dim)
                    teacher = masked_distribution(teacher_logits, visible[:, :, None]).mean(2)
                student = masked_distribution(scores, visible)
                kl = (
                    teacher * (teacher.clamp_min(1e-9).log() - student.clamp_min(1e-9).log())
                ).sum(-1)
                losses.append(kl.sum())
        candidates = torch.cat(pools, 1) if pools else (previous.candidates if previous else None)
        loss = sum(losses) / (b * s) if losses else zero
        return Selection(torch.cat(selections, 1), torch.cat(validities, 1), candidates), loss

    def forward(
        self,
        x,
        start=0,
        local=None,
        memory=None,
        previous=None,
        global_source=None,
        append_global=True,
        with_loss=True,
    ):
        c = self.c
        b, s, _ = x.shape
        positions = torch.arange(start, start + s, device=x.device)
        theta = 10000.0 if self.mode == "swa" else 40000.0
        qr = self.q_norm(self.q_a(x))
        q = rope(self.q_b(qr).view(b, s, c.heads, c.head_dim), positions, c.rope_dim, theta=theta)
        kv = rope(self.kv_norm(self.local_kv(x)), positions, c.rope_dim, theta=theta)
        if self.fake_quant:
            kv = fake_fp8(kv)
        local_start = start
        if local is not None:
            local_start = local.start
            kv = torch.cat((local.values, kv), 1)
        next_local = LocalMemory(kv[:, -c.window :], start + s - min(kv.size(1), c.window))
        if self.mode == "full" and append_global:
            source = x if global_source is None else global_source
            memory = self.extend_global(source, start, memory)
        selection = None
        index_loss = x.sum() * 0
        if self.mode != "swa":
            if memory is None:
                raise ValueError("Missing shared global KV")
            selection, index_loss = self.select(x, qr, q, positions, memory, previous, with_loss)
        outputs = []
        batch = torch.arange(b, device=x.device)[:, None, None]
        for lo in range(0, s, c.query_chunk):
            hi = min(s, lo + c.query_chunk)
            local_pos = positions[lo:hi, None] + torch.arange(1 - c.window, 1, device=x.device)
            local_valid = local_pos >= local_start
            local_idx = (local_pos - local_start).clamp(0, kv.size(1) - 1)
            selected = kv[:, local_idx]
            valid = local_valid[None].expand(b, -1, -1)
            if selection is not None and selection.indices.size(-1):
                global_values = memory.values[batch, selection.indices[:, lo:hi]]
                selected = torch.cat((selected, global_values), 2)
                valid = torch.cat((valid, selection.valid[:, lo:hi]), -1)
            scores = torch.einsum("bqhd,bqkd->bqhk", q[:, lo:hi], selected) / math.sqrt(c.head_dim)
            scores = scores.masked_fill(~valid[:, :, None], -torch.inf)
            sink = self.sink[None, None, :, None].expand(b, hi - lo, -1, -1)
            weights = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
            outputs.append(torch.einsum("bqhk,bqkd->bqhd", weights, selected))
        out = rope(torch.cat(outputs, 1), positions, c.rope_dim, inverse=True, theta=theta)
        out = out.reshape(b, s, c.output_groups, -1)
        out = torch.einsum("bsgd,grd->bsgr", out, self.out_a)
        return self.out_b(out.flatten(-2)), next_local, memory, selection, index_loss
