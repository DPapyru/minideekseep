import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            * self.weight
        ).to(x.dtype)


def rope(x, positions, rotary_dim, inverse=False, theta=10000.0):
    frequencies = theta ** (-torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim)
    angles = positions.float()[:, None] * frequencies[None, :]
    if inverse:
        angles = -angles
    shape = (1, len(positions)) + (1,) * (x.ndim - 3) + (rotary_dim // 2,)
    cos, sin = angles.cos().view(shape), angles.sin().view(shape)
    tail = x[..., -rotary_dim:].unflatten(-1, (-1, 2))
    a, b = tail[..., 0], tail[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)
    return torch.cat((x[..., :-rotary_dim], rotated.to(x.dtype)), -1)


class HyperConnection(nn.Module):
    """Each mixer computes coefficients once; pre weights feed the next sublayer."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.streams = n = c.hc_streams
        self.iters = c.sinkhorn_iters
        self.eps = c.norm_eps
        self.proj = nn.Linear(n * c.dim, (2 + n) * n, bias=False)
        self.base = nn.Parameter(torch.zeros((2 + n) * n))
        self.scales = nn.Parameter(torch.full((3,), 0.01))
        with torch.no_grad():
            self.base[2 * n :].view(n, n).copy_(torch.eye(n) * 4)

    def coefficients(self, residual):
        n = self.streams
        x = residual.flatten(-2).float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        p = self.proj(x)
        pre = torch.sigmoid(p[..., :n] * self.scales[0] + self.base[:n])
        post = 2 * torch.sigmoid(p[..., n : 2 * n] * self.scales[1] + self.base[n : 2 * n])
        mix = (p[..., 2 * n :] * self.scales[2] + self.base[2 * n :]).unflatten(-1, (n, n))
        mix = mix.softmax(-1)
        for _ in range(self.iters):
            mix = mix / mix.sum(-2, keepdim=True).clamp_min(self.eps)
            mix = mix / mix.sum(-1, keepdim=True).clamp_min(self.eps)
        return pre, post, mix

    @staticmethod
    def collapse(x, pre):
        return (x * pre.unsqueeze(-1)).sum(-2)

    @staticmethod
    def expand(y, residual, post, mix):
        return post.unsqueeze(-1) * y.unsqueeze(-2) + torch.einsum("bsij,bsjd->bsid", mix, residual)


class Expert(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.gate = nn.Linear(dim, width, bias=False)
        self.up = nn.Linear(dim, width, bias=False)
        self.down = nn.Linear(width, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.router = nn.Linear(c.dim, c.experts, bias=False)
        self.experts = nn.ModuleList([Expert(c.dim, c.expert_dim) for _ in range(c.experts)])
        self.shared = Expert(c.dim, c.expert_dim)
        self.top_k = c.top_k
        self.bias_rate = c.router_bias_rate
        self.register_buffer("route_bias", torch.zeros(c.experts))
        self.register_buffer("route_counts", torch.zeros(c.experts), persistent=False)

    def forward(self, x, collect_routes=False):
        flat = x.flatten(0, 1)
        scores = F.softplus(self.router(flat).float()).sqrt()
        indices = (scores + self.route_bias).topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        y = self.shared(flat)
        for i, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == i)
            # Empty expert inputs are legal and avoid GPU-to-CPU synchronization.
            contribution = expert(flat[tokens]) * weights[tokens, slots, None].to(flat.dtype)
            y = y.index_add(0, tokens, contribution)
        if collect_routes:
            with torch.no_grad():
                self.route_counts.add_(
                    torch.bincount(indices.flatten(), minlength=len(self.experts))
                )
        return y.view_as(x)

    @torch.no_grad()
    def update_bias(self):
        counts = self.route_counts
        if counts.sum() > 0:
            self.route_bias.add_(self.bias_rate * (counts.mean() - counts).sign())
            self.route_bias.sub_(self.route_bias.mean())
        counts.zero_()


class Engram(nn.Module):
    """Six independent n-gram tables, with a separate context gate for every HC branch."""

    def __init__(self, c):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(n, c.engram_head_dim) for n in c.engram_buckets])
        width = 6 * c.engram_head_dim
        self.key = nn.Linear(width, c.hc_streams * c.dim, bias=False)
        self.value = nn.Linear(width, c.dim, bias=False)
        self.key_weight = nn.Parameter(torch.ones(c.hc_streams, c.dim))
        self.query_weight = nn.Parameter(torch.ones(c.hc_streams, c.dim))
        self.dim = c.dim
        self.streams = c.hc_streams
        self.eps = c.norm_eps
        self.pad_id = c.pad_id

    def hashes(self, ids, history=None):
        b, s = ids.shape
        if history is None:
            history = ids.new_full((b, 3), self.pad_id)
        joined = torch.cat((history, ids), 1)
        result = []
        for table_idx, table in enumerate(self.tables):
            order = 2 + table_idx // 2
            value = torch.zeros_like(ids)
            for j in range(order):
                value = value * (257 + 2 * table_idx) + joined[:, 3 - j : 3 - j + s] + 1
                value = value.remainder(table.num_embeddings)
            result.append(value)
        return result, joined[:, -3:]

    def forward(self, residual, ids, history=None):
        hashes, new_history = self.hashes(ids, history)
        memory = torch.cat([t(h) for t, h in zip(self.tables, hashes)], -1)
        key = self.key(memory).unflatten(-1, (self.streams, self.dim)).float()
        query = residual.float()
        inverse_std = torch.rsqrt(query.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (query * self.query_weight * self.key_weight * key).sum(-1)
        dot = dot * inverse_std / math.sqrt(self.dim)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return residual + gate.unsqueeze(-1) * self.value(memory).unsqueeze(-2), new_history
