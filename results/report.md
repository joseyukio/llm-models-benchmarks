# Local LLM benchmark report

Generated **2026-10-19 14:16 -03** on `omarchy` - Omarchy, kernel `7.2.8-9-omarchy`.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 (12,227 MiB) |
| Driver | 615.78.08 |
| CPU | AMD Ryzen 5 5600X 6-Core Processor |
| RAM | 31.2 GiB |
| Backend | lmstudio - version: 0.7.0-dev (build 1, commit 2a66703); prism - version: 0.2.0-dev (build 10777, commit e3de71522) |

## At a glance

- Fastest generation: **Ornith-1.5-9B-Q6_K** - 78.5 t/s
- Fastest prefill: **Ornith-1.5-9B-Q6_K** - 3,635.9 t/s
- Largest fully-GPU context: **Ornith-1.5-9B-Q6_K** - 128k q8_0 at 10.1 GB total
- Models in report: 2 (2 speed, 2 context)

## Speed

Sorted by generation speed. Values are medians; min-max across repetitions in parentheses. VRAM is total used after load, desktop included.

| # | Model | Quant | Size | VRAM | Layers | Prefill t/s | Generation t/s | Backend |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Ornith-1.5-9B-Q6_K | Q6_K | 7.0 GB | 7.7 GB | all (ngl 99) | 3,635.9 (3,615.5-3,636.0) | 78.5 (78.4-78.6) | lmstudio |
| 2 | Ternary-Bonsai-2-27B-PQ2_0 | PQ2_0 | 6.7 GB | 8.4 GB | all (ngl 99) | 1,373.2 (1,360.6-1,379.3) | 66.6 (66.6-66.6) | prism |

## Context / VRAM map

Cell = total GPU memory after a successful load (desktop baseline included), or failure mode. `-` = skipped (beyond native context), `.` = not tested.

| Model | 32k f16 | 64k f16 | 64k q8_0 | 64k q4_0 | 128k q8_0 | 128k q4_0 |
|---|---:|---:|---:|---:|---:|---:|
| Ornith-1.5-9B-Q6_K | ✓ 8.4G | ✓ 9.5G | ✓ 8.7G | ✓ 8.2G | ✓ 10.1G | ✓ 9.1G |
| Ternary-Bonsai-2-27B-PQ2_0 | ✓ 9.9G | ✗ OOM | ✓ 10.0G | ✓ 9.0G | ✗ OOM | ✓ 10.2G |

## Context load failures

| Model | Rung | Error | Log |
|---|---|---|---|
| Ternary-Bonsai-2-27B-PQ2_0 | 64k f16 | OOM | `results/logs/20261019-141428-ternary-bonsai-2-27b-pq2_0-c65536-f16-52943.log` |
| Ternary-Bonsai-2-27B-PQ2_0 | 128k q8_0 | OOM | `results/logs/20261019-141442-ternary-bonsai-2-27b-pq2_0-c131072-q8_0-60979.log` |

## Model notes

### Ornith-1.5-9B-Q6_K

`/run/media/yukio/SSD-1_Tera/common-home/yukio/llm-models/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q6_K.gguf`

- Historically the fastest model on this machine: ~77.4 t/s generation and
  ~3,353 t/s prefill in the manual harness.
- Best pure prefill performance of the current lineup; well suited to agent
  loops and repeated long-prompt work.
- Strong software-engineering benchmark results for its size (best of the 9B
  class in the manual comparison).
- Fully GPU-resident at 8k (about 7.6 GB total). 128k with q8_0 KV previously
  measured at ~10.1 GB total; 192k is too tight.

### Ternary-Bonsai-2-27B-PQ2_0

`/run/media/yukio/SSD-1_Tera/common-home/yukio/llm-models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf`

- 27B-class model at 1.72 bits/weight (7.2 GB); retains ~98% of the FP16
  model's benchmark score according to the publisher.
- **Requires the Prism ML llama.cpp fork.** Stock llama.cpp, LM Studio and
  Ollama cannot load PQ2_0/PTQ1_0 correctly. The `prism` backend profile and
  filename `match` rules handle this automatically.
- Manual measurements on this machine: ~67.0 t/s generation (512-token sample)
  and ~1,356 t/s warm prefill; 8,406 MiB total at 8k. The first request after a
  cold start can be several times slower - llmbench discards warm-up.
- Context ceiling: 32k f16 ~9.9 GB; 64k q8_0 ~10.1 GB; 128k q4_0 ~10.3 GB.
  64k f16 and 128k q8_0 do not fit on 12 GB.
- Thinking is enabled by default (xhigh effort). Use `medium` effort for
  everyday chat; recommended sampling is temp 1.0, top_p 0.95, top_k 20,
  min_p 0.05.
- Optional vision projector (mmproj Q8_0, ~0.63 GB) adds image input.

## Methodology

- Backend: `llama-server`, one slot, `-ngl 99`, `-fa on`, `-b 2048 -ub 512`, `--parallel 1`, no speculative decoding.
- Speed: warm-up requests discarded, then 3 repetitions with ~1201 prompt tokens and exactly 128 generated tokens (`ignore_eos`), temperature 0, `cache_prompt: false`. Timings are server-reported; median with min-max.
- VRAM: total used reported by `nvidia-smi` after load, including the desktop baseline.
- Context: successful server startup at each ladder rung; KV cache type as labeled. Failures are load failures, usually OOM.
- Fully-GPU only: models that require CPU offload are out of scope. Expect roughly +/-5% run-to-run variation.

## Raw data

- `results/<slug>/speed.json`, `results/<slug>/context.json`, `results/<slug>/sanity.json`
- Server logs: `results/logs/` (gitignored)
