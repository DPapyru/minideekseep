# Architecture and Counting

This project studies DeepSeek V4.1's design at a much smaller text-only scale.
The upstream source is pinned at `314113f8a7dadb388215396ba5fbee2c610a04d6`.
See `THIRD_PARTY_NOTICES.md` for the upstream MIT notice.

## Source Mapping

| Upstream reference | Local adaptation |
| --- | --- |
| `inference/model.py`: Attention, Compressor, Indexer | `attention.py`: differentiable projections, complete-group compression, sparse selection and explicit state |
| `inference/model.py`: Gate, Expert, MoE | `layers.py`: sqrt-softplus, selection correction bias, two routed and one shared expert, no token dropping |
| `inference/model.py`: Block / HC functions | `layers.py` / `model.py`: delayed pre-mix, bounded pre/post coefficients and Sinkhorn residual mixing |
| `inference/model.py`: Engram | `layers.py`: branch-specific keys and signed-square-root normalized gate |
| Technical report sections 2.2 and 3.2.2 | `model.py`: encoder-derived decoder global KV and approximate bounded decoder replay |
| DSpark reference and report section 2.4.3 | `dspark.py`: parallel base positions, Markov dependence, confidence survival, frozen-backbone draft stage and rejection correction |

The upstream inference forward is decorated with inference mode and uses specialized
low-precision kernels. It cannot become a trainable SM75 implementation by changing
only its shapes. This project uses autograd-compatible PyTorch operations, explicit
per-request caches and no custom CUDA dependency. The source checkout is retained
for inspection rather than imported as a runtime dependency.

## Main Configuration

12 blocks, width 512, Q rank 256, eight 64-dimensional query heads sharing one KV
latent, four output projection groups of rank 128. There are eight routed SwiGLU
experts of intermediate width 736, with two selected per token and one shared expert.
The four HC streams each retain width 512; they are not four independent models.

The fixed schedule, numbered from zero, is:

| Layer | Stage | Global mode | Global source |
| --- | --- | --- | --- |
| 0 | Encoder | SWA only | None |
| 1, 3 | Encoder | Full, ratio 2 | Current normalized attention input |
| 2 | Encoder | Reuse | Layer 1 KV and indices |
| 4 | Encoder | Reindex | Layer 3 KV, new indices |
| 5 | Encoder | Reuse | Layer 3 KV, layer 4 indices |
| 6 | Decoder | Full, ratio 1 | Final encoder state, collapsed and normalized |
| 7, 9, 11 | Decoder | Reuse | Decoder KV and preceding selection |
| 8, 10 | Decoder | Reindex | Decoder KV; score only candidates from layer 6 |

Every block generates its own local SWA KV. The first decoder Full layer creates
the candidate pool: at most sixteen blocks of four global entries. Later decoder
indexers gather that pool before scoring it. The small reference still scans and
sorts the pool mask; it is not a production constant-cost indexing kernel.

Training uses full-sequence causal cross-entropy plus a 0.01-weighted indexer KL.
The indexer teacher is the detached mean-over-heads global attention distribution;
for hierarchical reindexing it is restricted to the candidate pool. This is a
small-model training choice rather than a reproduction of DeepSeek's training code.
The teacher permits learning outside the main attention's discrete selected set.

## Counts

The default model has these exact unique parameter counts:

| Group | Parameters |
| --- | ---: |
| Encoder active backbone, excluding Engram | 24,947,124 |
| Encoder + decoder active backbone, excluding Engram | 49,793,800 |
| Stored backbone including all experts | 131,189,512 |
| Input embedding | 4,194,304 |
| Output head, untied | 4,194,304 |
| Engram tables | 1,583,232 |
| Engram projections and gates | 991,232 |
| Final RMSNorm | 512 |
| Total, before DSpark | 142,153,096 |

Activation counts exclude inactive routed experts, not merely untouched memory.
The reference dispatch can invoke empty expert tensors; its launch overhead is
measured in throughput but does not add useful active expert computation.

Prefill also projects decoder global KV and executes decoder SWA replay over the
last 32 positions. Its total work is not represented by the 25M encoder figure.
Training updates the full stored model and always executes both halves. FP32
parameters, gradients and two Adam moments alone occupy about 2.118GiB; caches,
activations and allocator/workspace overhead are additional.

## Deliberate Differences

- 8192-token custom BPE; raw token n-gram hashing instead of the upstream normalized,
  compressed vocabulary and layer-seeded XOR multipliers.
- Small independent prime-sized Engram tables, with six lookups per token/module.
- FP32 AdamW rather than the official Muon, Sinkhorn-balanced embedding updates and
  FP8/FP4 weight paths. RMSNorm epsilon is 1e-6 for FP32 numerical stability.
- No vision encoder, distributed training, million-token extrapolation or post-training RL.
- DSpark is a later, separately stored draft. Its sequential verifier is a correctness
  reference and is disabled by default; it does not promise throughput improvement.
- Fake FP4/FP8 cache experiments preserve floating-point storage and use software
  rounding/straight-through gradients. No packed cache or native low-precision speedup.

Exact cached decoding and bounded replay are different interfaces. The latter
truncates decoder local history during prefill and marks the resulting cache
approximate. Tests compare exact decoding against a full forward; approximate
replay is evaluated by loss and timing, not by asserting identical hidden states.
