from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import GlobalMemory, LocalMemory, SparseAttention
from .config import ModelConfig
from .layers import Engram, HyperConnection, MoE, RMSNorm


@dataclass
class KVCache:
    position: int = 0
    local: dict[int, LocalMemory] = field(default_factory=dict)
    global_kv: dict[int, GlobalMemory] = field(default_factory=dict)
    histories: dict[int, torch.Tensor] = field(default_factory=dict)
    approximate: bool = False


@dataclass
class TrainOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    index_loss: torch.Tensor
    logits: torch.Tensor | None = None


class Block(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.attention = SparseAttention(c, index)
        self.moe = MoE(c)
        self.attn_hc = HyperConnection(c)
        self.ffn_hc = HyperConnection(c)
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        self.engram = Engram(c) if index in c.engram_layers else None

    def forward(
        self,
        h,
        pre,
        ids,
        start,
        local,
        memory,
        previous,
        global_source,
        append_global,
        history,
        with_loss,
    ):
        if self.engram is not None:
            h, history = self.engram(h, ids, history)
        attn_pre, attn_post, attn_mix = self.attn_hc.coefficients(h)
        x = self.attn_norm(self.attn_hc.collapse(h, pre))
        y, local, memory, previous, aux = self.attention(
            x, start, local, memory, previous, global_source, append_global, with_loss
        )
        h = self.attn_hc.expand(y, h, attn_post, attn_mix)
        next_pre, ffn_post, ffn_mix = self.ffn_hc.coefficients(h)
        x = self.ffn_norm(self.ffn_hc.collapse(h, attn_pre))
        y = self.moe(x)
        return (
            self.ffn_hc.expand(y, h, ffn_post, ffn_mix),
            next_pre,
            local,
            memory,
            previous,
            aux,
            history,
        )


class MiniDeepSeek(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = c = config
        self.embedding = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList([Block(c, i) for i in range(len(c.modes))])
        self.norm = RMSNorm(c.dim, c.norm_eps)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.apply(self._init)
        # Small initial residual updates make the twelve-layer HC stack trainable from scratch.
        for block in self.layers:
            nn.init.normal_(block.attention.out_b.weight, std=0.02 / (2 * len(self.layers)) ** 0.5)
            for expert in [block.moe.shared, *block.moe.experts]:
                nn.init.normal_(expert.down.weight, std=0.02 / (2 * len(self.layers)) ** 0.5)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def _run(self, ids, cache=None, bounded=False):
        c = self.config
        b, s = ids.shape
        start = cache.position if cache else 0
        if s < 1 or start + s > c.max_seq_len:
            raise ValueError(f"Sequence must be nonempty and fit within {c.max_seq_len} tokens")
        h = self.embedding(ids).unsqueeze(2).expand(-1, -1, c.hc_streams, -1)
        pre = h.new_zeros(b, s, c.hc_streams)
        pre[..., 0] = 1
        memory = selection = encoder_h = None
        source_id = None
        aux = h.sum() * 0
        new_cache = KVCache(
            position=start + s, approximate=bounded or bool(cache and cache.approximate)
        )
        index_sources = 0
        for i, layer in enumerate(self.layers):
            if i == c.encoder_layers:
                encoder_h = layer.attn_norm(layer.attn_hc.collapse(h, pre))
                if bounded:
                    # Decoder replay keeps absolute positions and sees the complete encoder KV.
                    replay = min(s, c.window)
                    h, pre = h[:, -replay:], pre[:, -replay:]
                memory = selection = None
            mode = c.modes[i]
            layer_start = start + s - h.size(1)
            local = cache.local.get(i) if cache else None
            history = cache.histories.get(i) if cache else None
            if mode == "full":
                source_id = i
                memory = cache.global_kv.get(i) if cache else None
                selection = None
            source = encoder_h if i == c.encoder_layers else None
            append = True
            if bounded and i == c.encoder_layers:
                memory = layer.attention.extend_global(source, start, memory)
                append = False
            layer_ids = ids[:, -h.size(1) :]
            inputs = (
                h,
                pre,
                layer_ids,
                layer_start,
                local,
                memory,
                selection,
                source,
                append,
                history,
                self.training,
            )
            if self.training and c.checkpoint_layers:
                # Non-reentrant checkpoint preserves gradients through shared KV dataclasses.
                result = checkpoint(layer, *inputs, use_reentrant=False)
            else:
                result = layer(*inputs)
            h, pre, local, memory, selection, loss, history = result
            aux = aux + loss
            index_sources += int(mode in ("full", "reindex"))
            new_cache.local[i] = local
            if mode == "full":
                new_cache.global_kv[source_id] = memory
            if history is not None:
                new_cache.histories[i] = history
        hidden = self.norm(HyperConnection.collapse(h, pre))
        return hidden, aux / max(index_sources, 1), new_cache

    def forward(self, ids):
        hidden, _, _ = self._run(ids)
        return self.head(hidden)

    def forward_train(self, input_ids, labels, return_logits=False):
        hidden, index_loss, _ = self._run(input_ids)
        # Chunking the output head prevents the vocabulary dimension from dominating peak memory.
        total = hidden.sum() * 0
        valid = (labels != -100).sum().clamp_min(1)
        logits_list = []
        for lo in range(0, hidden.size(1), self.config.query_chunk):
            h, y = (
                hidden[:, lo : lo + self.config.query_chunk],
                labels[:, lo : lo + self.config.query_chunk],
            )

            def chunk_loss(features, targets):
                logits = self.head(features)
                return F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum"
                )

            if self.training and self.config.checkpoint_layers:
                total = total + checkpoint(chunk_loss, h, y, use_reentrant=False)
            else:
                total = total + chunk_loss(h, y)
            if return_logits:
                logits_list.append(self.head(h))
        lm_loss = total / valid
        return TrainOutput(
            lm_loss + self.config.index_loss_weight * index_loss,
            lm_loss,
            index_loss,
            torch.cat(logits_list, 1) if return_logits else None,
        )

    def forward_replay(self, input_ids, labels):
        """Train on bounded-replay prompts, supervising their final next-token prediction."""
        hidden, index_loss, _ = self._run(input_ids, bounded=True)
        lm_loss = F.cross_entropy(self.head(hidden[:, -1]), labels[:, -1])
        return TrainOutput(
            lm_loss + self.config.index_loss_weight * index_loss, lm_loss, index_loss
        )

    def set_fake_quantization(self, enabled=True):
        for layer in self.layers:
            layer.attention.fake_quant = enabled

    @torch.no_grad()
    def prefill(self, input_ids, bounded=False):
        if self.training:
            raise ValueError("Call eval() before prefill")
        hidden, _, cache = self._run(input_ids, bounded=bounded)
        return self.head(hidden[:, -1]), cache

    @torch.no_grad()
    def decode_step(self, input_ids, cache):
        if self.training or input_ids.size(1) != 1:
            raise ValueError("decode_step requires eval mode and one token per batch")
        hidden, _, cache = self._run(input_ids, cache=cache)
        return self.head(hidden[:, -1]), cache

    @torch.no_grad()
    def update_router_biases(self):
        for layer in self.layers:
            layer.moe.update_bias()

    def parameter_report(self):
        c = self.config
        encoder = decoder = stored = 0
        for i, layer in enumerate(self.layers):
            all_params = sum(p.numel() for p in layer.parameters())
            engram = sum(p.numel() for p in layer.engram.parameters()) if layer.engram else 0
            expert = sum(p.numel() for p in layer.moe.experts[0].parameters())
            active = all_params - engram - (c.experts - c.top_k) * expert
            if i < c.encoder_layers:
                encoder += active
            else:
                decoder += active
            stored += all_params - engram
        tables = sum(
            sum(p.numel() for p in layer.engram.tables.parameters())
            for layer in self.layers
            if layer.engram
        )
        engram_all = sum(
            sum(p.numel() for p in layer.engram.parameters())
            for layer in self.layers
            if layer.engram
        )
        total = sum(p.numel() for p in self.parameters())
        return dict(
            encoder_active=encoder,
            decode_active=encoder + decoder,
            stored_backbone=stored,
            embedding=self.embedding.weight.numel(),
            output_head=self.head.weight.numel(),
            engram_tables=tables,
            engram_compute=engram_all - tables,
            final_norm=sum(p.numel() for p in self.norm.parameters()),
            total=total,
            fp32_adam_state_gib=total * 16 / 2**30,
            encoder_target_met=abs(encoder / 25e6 - 1) <= 0.05,
            decode_target_met=abs((encoder + decoder) / 50e6 - 1) <= 0.05,
        )
