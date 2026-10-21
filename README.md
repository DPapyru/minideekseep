# MiniDeepSeek V4.1

A from-scratch, text-only architecture study for a GTX 1650 4GB. This is not an
official DeepSeek model, and it does not load DeepSeek's released weights.

See [architecture and parameter counts](docs/architecture.md) for the upstream
source mapping, fixed layer schedule and deliberate small-model adaptations.
See [local verification](docs/verification.md) for the installed environment,
real-text smoke run, GPU memory measurements and remaining training stages.

The target is approximately 25M active backbone parameters in the encoder and
50M across the encoder plus decoder. Stored experts, embeddings, the output head,
Engram tables and the later DSpark drafter make the stored model larger. Run
`params` for exact counts. Prefill also projects decoder global KV and replays a
short decoder window; the 25M figure does not account for all of that work.
Pretraining runs the entire encoder and decoder.

The current default is **24,947,124 encoder-active / 49,793,800 decode-active**
parameters, with **142,153,096 total stored parameters** before DSpark.

## Install

```bash
uv sync --python 3.11
uv run minideekseep params --output runs/parameters.json --save-config runs/model.json
uv run pytest
uv run ruff check .
```

The lockfile uses PyTorch 2.14 with CUDA 12.6 wheels. The NVIDIA driver can be
newer; the locally installed `nvcc` is not used. No custom CUDA extension,
FlashMLA, Tensor Core, BF16, or native FP8/FP4 hardware support is required.
The CUDA wheel must include `sm_75` support. Commands default to four CPU threads.

If large downloads repeatedly restart, `tools/download_wheels.py PACKAGE` downloads
one of the locked NVIDIA wheels in resumable ranges and verifies its SHA256. This
is a connection recovery tool, not a different dependency source. Completed wheels
can be installed with `uv pip install --no-deps data/wheels/NAME.whl`, followed by
`uv sync --frozen` to check the complete environment.

## Data

```bash
uv run minideekseep prepare --documents 30000
```

For a small connectivity/engineering pilot without reading a gigabyte Parquet
shard, use `prepare --data data/preview --documents 600 --preview`. The preview API
serves the current dataset, not a revision-pinned download; its current revision is
checked before and after collection and the exact local records are hashed.

This streams a bounded sample of FineWiki `zh` and `en`, resolves and records its
revision, normalizes NFC/line endings, deduplicates normalized text, and splits
documents by content hash into 98/1/1 train/validation/test. The byte-level BPE
tokenizer is trained only on the training split. Metadata records the Wikipedia
URLs and the source license, CC BY-SA 4.0 / GFDL. Dataset collection order is not
treated as a random global sample. Inspect the source and encoded manifests before
a long run. For more data, prepare a new directory with a larger document limit.

Custom corpora can be supplied as JSONL containing `text` and optional
`language` (`zh` or `en`), `id`, `url`, and `title`:

```bash
uv run minideekseep prepare --data data/custom --source /path/to/corpus.jsonl
```

The encoded corpus contains uint16 token files and document offsets. Every sampled
sequence belongs to one document, so attention/Engram state cannot cross document
boundaries. Short documents are padded with ignored targets. The resumable sampler
mixes approximately 95% Chinese and 5% English by real token count. Exact duplicate
documents cannot cross splits; near duplicates are not removed. Each segment starts
with empty history, including segments sampled from the middle of a document.

## Measure Before Training

```bash
uv run minideekseep benchmark --length 128
uv run minideekseep benchmark --length 256 --output runs/benchmark-256.json
```

The default benchmark runs at least 100 optimizer steps and ten minutes, including
gradient accumulation to a nominal 4096 tokens per step. It records peak allocated
and reserved GPU memory and an estimated 72-hour token budget capped at 50M.
If process reserved memory exceeds 2.6GiB, the run stops. Retry length 128 if 256
does not fit. Model dimensions never change automatically. Free GPU memory also
depends on desktop programs; a benchmark may fail before reaching this threshold.

For a short implementation smoke check, explicitly reduce the work:

```bash
uv run minideekseep benchmark --steps 1 --seconds 0 --tokens-per-step 128
```

## Train, Resume, Evaluate

```bash
uv run minideekseep train --length 128 --tokens 1000000 --run runs/pilot
uv run minideekseep train --length 128 --tokens 1000000 --run runs/pilot --resume runs/pilot/latest.pt
uv run minideekseep evaluate runs/pilot/latest.pt --length 128
uv run minideekseep generate runs/pilot/latest.pt --prompt "中国的首都是"
```

Use the measured budget for a separate formal run after the pilot is stable.
FP32 AdamW uses LR 3e-4, 2% warmup, cosine decay to 3e-5, clipping at 1.0,
and layer/output-head gradient checkpointing. Route correction biases update once
per optimizer step. Metrics are JSONL. Validation runs every 100 steps. Checkpoints
are saved every 15 minutes and at normal completion; the latest two periodic
checkpoints and best validation checkpoint are retained. Interruptions resume from
the last saved completed step, not an unfinished gradient accumulation.

Resume requires identical model config, tokenizer, encoded manifest, sequence
length, LR, token budget and accumulation settings. Checkpoints include all RNG
states, optimizer, route biases and data sampler state. Only load trusted `.pt`
files: optimizer/RNG checkpoints use Python pickle. Storage for an FP32 model plus
Adam states is roughly 1.6GB per checkpoint; retaining three needs about 5GB.

## Later Stages

```bash
uv run minideekseep train --initialize runs/pilot/latest.pt --replay --tokens 1000000 --run runs/replay
uv run minideekseep generate runs/replay/latest.pt --bounded --prompt "中国的首都是"
uv run minideekseep evaluate runs/replay/latest.pt --compare-prefill
uv run minideekseep train-draft runs/replay/latest.pt --run runs/dspark
uv run minideekseep profile-draft runs/dspark/draft.pt --prompt "中国的首都是"
uv run minideekseep generate-draft runs/dspark/draft.pt --prompt "中国的首都是"
uv run minideekseep train --initialize runs/replay/latest.pt --fake-quant --tokens 1000000 --run runs/qat
```

Replay adaptation supervises next-token prediction after a prompt whose decoder
sees only the final sliding window. It retains complete encoder-derived global KV.
This approximation is explicitly separate from exact cached decoding. It needs its
own held-out loss comparison; no equality or speedup is assumed.

DSpark has two small SWA/MoE blocks, a three-position parallel base prediction,
a rank-32 autoregressive Markov head and an acceptance confidence head. The frozen
backbone supplies features and a shared embedding/output head. A separate on-policy
distillation stage trains the draft. The reference rejection verifier applies the
positive `p-q` residual and functional cache rollback; verification is sequential,
so it is a correctness reference rather than a high-throughput engine. Its profile
chooses a verification length from measured curves; it stays disabled by default.
Draft checkpoints save every 15 minutes and at completion. Resume the same stage
with `train-draft ... --resume runs/dspark/draft.pt`, keeping its token budget,
sequence length, width and backbone checkpoint identical.
`generate-draft` currently re-encodes the growing prefix for each round. Use it to
validate sampling and draft behavior; it is not the optimized serving path.

QAT simulates E2M1 FP4 global KV, E4M3 group scales, FP4 indexer Q/K and FP8 local
KV through straight-through gradients. These values remain in floating-point
storage; this experiment does not reduce storage or accelerate the 1650. It is
not a packed quantized deployment format. Evaluate its quality separately.

## Architecture and Upstream

The upstream source is cloned in `references/DeepSeek-V4.1-Flash`; the fixed
revision and MIT notice are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
To recreate the reference checkout without downloading model weights:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash references/DeepSeek-V4.1-Flash
git -C references/DeepSeek-V4.1-Flash sparse-checkout set inference encoding evaluation
```

The trainable implementation retains CED encoder-source decoder KV, Full/Reindex/
Reuse, learned gated compression, hierarchical candidate blocks, shared-latent
attention, shared+routed experts, delayed Single-Pass mHC coefficients and
branch-specific signed-square-root Engram gating. Indexers learn from a detached
dense global-attention teacher. It uses a much smaller tokenizer and raw token
n-gram hashes instead of the upstream compressed-token hash vocabulary. Optimization
uses AdamW rather than Muon/Sinkhorn, and floating-point operations rather than
the upstream low-precision inference kernels. It is not checkpoint-compatible.

Tests exercise causality, complete-block visibility, exact cached decoding through
window wraps, approximate replay, shared KV/indexer/expert/Engram gradients,
checkpoint equivalence, HC constraints, overfitting, tokenizer byte roundtrips,
sampler/checkpoint restoration, fake quantization and speculative correction.
The 50M activated model is a small language-model experiment: useful Chinese
conversation, long-context extrapolation and frontier-model capabilities are not
acceptance criteria. No web UI, API server or pretrained weights are included.
