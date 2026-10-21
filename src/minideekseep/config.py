import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    experts: int = 8
    top_k: int = 2
    expert_dim: int = 736
    heads: int = 8
    head_dim: int = 64
    rope_dim: int = 16
    q_rank: int = 256
    output_groups: int = 4
    output_rank: int = 128
    window: int = 32
    encoder_compression: int = 2
    index_heads: int = 4
    index_dim: int = 32
    index_top_k: int = 16
    candidate_blocks: int = 16
    candidate_block_size: int = 4
    hc_streams: int = 4
    sinkhorn_iters: int = 20
    engram_layers: tuple[int, ...] = (1, 4)
    engram_buckets: tuple[int, ...] = (4099, 4111, 4127, 4129, 4133, 4139)
    engram_head_dim: int = 32
    max_seq_len: int = 1024
    query_chunk: int = 32
    norm_eps: float = 1e-6
    index_loss_weight: float = 0.01
    router_bias_rate: float = 0.001
    checkpoint_layers: bool = True
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    noise_id: int = 4

    def __post_init__(self):
        if self.encoder_layers != 6 or self.decoder_layers != 6:
            raise ValueError("The fixed CSA2 schedule requires six encoder and six decoder layers")
        if not 0 < self.top_k <= self.experts:
            raise ValueError("Invalid expert selection count")
        if self.heads % self.output_groups or self.rope_dim % 2:
            raise ValueError("Invalid output grouping or rotary dimension")
        if self.rope_dim > min(self.head_dim, self.index_dim):
            raise ValueError("Rotary dimension exceeds attention/index head dimension")
        if len(self.engram_buckets) != 6 or self.encoder_compression < 1:
            raise ValueError("Engram requires two tables for each of n=2,3,4")
        if min(self.dim, self.window, self.query_chunk, self.index_top_k) < 1:
            raise ValueError("Dimensions and window sizes must be positive")
        if self.vocab_size <= self.noise_id or self.noise_id != 4:
            raise ValueError("Tokenizer special IDs must match the fixed five-token prefix")
        if (
            min(
                self.index_heads,
                self.index_dim,
                self.hc_streams,
                self.sinkhorn_iters,
                self.candidate_blocks,
                self.candidate_block_size,
                self.max_seq_len,
            )
            < 1
        ):
            raise ValueError("Attention and residual dimensions must be positive")

    @property
    def modes(self):
        return (
            "swa",
            "full",
            "reuse",
            "full",
            "reindex",
            "reuse",
            "full",
            "reuse",
            "reindex",
            "reuse",
            "reindex",
            "reuse",
        )

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        for key in ("engram_layers", "engram_buckets"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def tiny(cls, **overrides):
        values = dict(
            vocab_size=288,
            dim=32,
            experts=4,
            top_k=2,
            expert_dim=24,
            heads=2,
            head_dim=8,
            rope_dim=4,
            q_rank=16,
            output_groups=2,
            output_rank=8,
            window=4,
            index_heads=2,
            index_dim=8,
            index_top_k=2,
            candidate_blocks=2,
            candidate_block_size=2,
            engram_buckets=(17, 19, 23, 29, 31, 37),
            engram_head_dim=4,
            max_seq_len=128,
            query_chunk=4,
            checkpoint_layers=False,
        )
        values.update(overrides)
        return cls(**values)
