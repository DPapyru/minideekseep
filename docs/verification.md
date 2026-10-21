# Local Verification

Measured on 2026-10-21 in this workspace. These are engineering checks of a small
architecture implementation, not a reproduction of DeepSeek's model quality.

## Environment

- NVIDIA GeForce GTX 1650, 4096MiB, compute capability 7.5.
- NVIDIA driver 595.104.02; PyTorch 2.14.0+cu126, Python 3.11.17.
- The installed wheel includes `sm_75`. CUDA forward/backward works.
- `uv sync --frozen --offline`, `uv pip check`, and offline wheel/sdist builds pass.
- Upstream source is cloned, without weights, at
  `references/DeepSeek-V4.1-Flash`, revision
  `314113f8a7dadb388215396ba5fbee2c610a04d6`.

## Final Checks

`uv run --offline pytest -q`: **27 passed in 19.71 seconds**, including the CUDA
training/cache/replay-QAT test on the GTX 1650. Coverage includes causal masks,
compression boundaries, window wraps, shared KV/indexer/Engram/router gradients,
checkpointed-gradient equivalence, finite Sinkhorn constraints, overfitting,
document boundaries, data integrity, resumable sampling/training/drafts, speculative
rejection/cache isolation, and budget rounding. Ruff and dependency checks pass.

## Hardware Smoke Measurements

Both measurements use the full 142,153,096-parameter model, FP32 AdamW,
gradient checkpointing, and one optimizer step. They include initialization/first
step effects and are not stable throughput estimates.

| Sequence | Tokens/step | Allocated peak | Reserved peak | First-step time |
| --- | ---: | ---: | ---: | ---: |
| 128 | 128 | 2218.72MiB | 2358MiB | 1.392s |
| 256 | 256 | 2229.60MiB | 2374MiB | 1.638s |

Raw results are in `runs/benchmark-gpu-smoke.json` and
`runs/benchmark-gpu-256-smoke.json`.

## Sustained GTX 1650 Benchmark

The full model completed **100 optimizer steps**, sequence length **256**, with
gradient accumulation to **4096 tokens per step**: 409,600 synthetic target tokens.

| Metric | Measured result |
| --- | ---: |
| Elapsed time | 1605.66s (26m 46s) |
| Throughput | 255.10 token/s |
| Peak allocated GPU memory | 2278.69MiB |
| Peak reserved GPU memory | 2442MiB |
| Approximate 1M-token time at this speed | 65.3 minutes |
| Approximate 50M-token time at this speed | 54.4 hours |
| Suggested 72-hour budget, capped and step-aligned | 49,999,872 tokens |

The process remained below the 2.6GiB reserved-memory limit. Desktop programs were
also running; total device usage observed during training was approximately 3.2GiB.
This repeated fixed random-sequence benchmark demonstrates stability and measures
compute throughput. Its declining loss is memorization, not language capability.

Raw measurements are in `runs/benchmark-gpu-256.json`, with per-step logs in
`runs/benchmark-gpu-256.log`. The process started before the budget-rounding fix;
its log prints the old 50,000,000 cap. The JSON's suggested budget was corrected to
49,999,872 (12,207 full steps), without changing any timing or memory measurements.
The current code applies the cap before rounding by actual accumulated tokens.

## Real-Text Pilot

`data/preview-600` contains 600 Chinese and 100 English FineWiki records, collected
through its dataset viewer. It is an ordered small sample for connectivity and
pipeline verification, not a representative pretraining corpus. It is skewed
toward sports/encyclopedia topics. Observed source revision:
`8bd13e72e6a002407649b3e898535f42ceb1aeb9`.

The viewer serves current data and may itself be cached; the observed revision is
not a guarantee of revision-pinned retrieval. The exact collected records and
encoded files are hashed locally. Use the normal revision-pinned streaming path
for a larger corpus when network access to its gigabyte Parquet shards is reliable.

- Document split: 682 train, 8 validation, 10 test.
- Encoded training tokens: 494,361 Chinese + 109,201 English.
- Trained vocabulary: 8192; byte-level fallback roundtrips Chinese and emoji.
- Full model completed 100 CPU optimizer steps, including checkpoint restoration.
- Real training targets processed: 9,516 tokens. This smoke run used the earlier
  fixed-microbatch schedule; current training accumulates at least the requested
  real target-token count per optimizer step.
- Validation loss at step 100: 8.028626 nats/token on 898 sampled validation targets.
- `runs/pilot-smoke/latest.pt` is usable for generation and resuming that exact
  smoke configuration. It is not a useful pretrained assistant.

Example generation after this very short training was incoherent and reflected
the corpus bias. No capability score or useful Chinese question-answering ability
is claimed.

Reproduce evaluation or inspect generated text:

```bash
uv run minideekseep evaluate runs/pilot-smoke/latest.pt --data data/preview-600 --length 128 --split validation
uv run minideekseep generate runs/pilot-smoke/latest.pt --tokenizer data/preview-600/tokenizer.json --prompt "中国的首都是"
```

## Replay and Draft Checks

On ten sampled validation prompts, CPU exact/bounded next-token losses were
4.394893 / 4.393726, with average times 0.12245 / 0.10331 seconds. These are prompt
end-point losses, not the all-token loss above. The small sample and short pilot
do not establish a general quality/speed tradeoff. Raw comparison:
`runs/pilot-smoke/replay-comparison.json`.

The full draft configuration completed one CPU distillation step and saved
`runs/draft-smoke/draft.pt`. Profile and generation commands successfully reload
its bound backbone. One CPU profile round measured a draft/reference speed ratio
of 0.802, so speculative decoding remains disabled by default. This mostly
untrained draft and one-round timing are only workflow smoke checks. Raw profile:
`runs/draft-smoke/profile.json`.

## What Remains Experimental

The 1M-token pilot, replay adaptation, 1M-token draft training, QAT adaptation,
and a multi-day 50M-token pretraining run have not been completed. They are separate
training stages provided by the CLI. No large training job is automatically left
running after implementation verification.

Before a formal run, collect a larger, more representative corpus and free disk
space. A full FP32 model/Adam checkpoint is about 1.6GB; the latest two plus best
checkpoint require about 5GB, with additional transient space during atomic saves.
Reserve at least 8GB free for a new run, beyond the dataset and environment.

The existing small corpus can exercise a new 1M-token pilot after freeing space:

```bash
uv run minideekseep train --data data/preview-600 --run runs/pilot-1m --length 256 --tokens 1000000
```

It will repeat this small, biased corpus. Collect a larger corpus for meaningful
pretraining; this command validates the training workflow rather than promising
model quality. No multi-hour training is started automatically.

Synthetic benchmark speed is an estimate. Short real documents, expert routing,
validation, checkpoint I/O, desktop workload and storage can reduce throughput.
The formal token budget is capped at 50M and rounded to optimizer-step multiples;
it must be chosen from measured throughput, not from GPU model-name estimates.
