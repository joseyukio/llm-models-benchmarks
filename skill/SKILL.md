---
name: LLM Benchmarks
description: Benchmark local GGUF models on this machine (speed, VRAM, context) with the llm-model-benchmarks repo and produce the Markdown report. Use when asked to benchmark a model, compare local models, check whether a model fits in VRAM, measure tokens/s, map context limits, or refresh the local LLM benchmark report.
---

# LLM benchmarks

Tool: `/home/yukio/dev/llm-models-benchmarks/llmbench.py` (on PATH as
`llmbench`). Config: `config.toml` in the same repo. Results:
`results/<slug>/`, report: `results/report.md`, qualitative notes:
`notes/<slug>.md`.

## Preconditions

- 12 GB RTX 5070: only benchmark when idle. Check `nvidia-smi`; desktop baseline
  should be under ~1.5 GB. If LM Studio has a model loaded, ask the user to
  unload it first - otherwise every VRAM number is wrong.
- Never hand-roll a `llama-server` command for benchmarking. Use the configured
  backend profiles so flags and CUDA paths match previous runs.
- Fully-GPU only (`-ngl 99`). Do not benchmark CPU-offloaded configurations;
  they are out of scope and not comparable.

## Workflow

1. Discover: `llmbench list` (or `list --json`).
2. New model: put the GGUF under a configured `model_roots` directory, then
   `llmbench list`.
3. Speed: `llmbench bench <model>` (8k f16, 3 reps, warm-up discarded).
4. Context: `llmbench context <model>` (ladder in `config.toml`).
5. Sanity: `llmbench sanity <model>` - verifies the quant produces coherent
   output, not garbage.
6. Notes: add `notes/<slug>.md` with qualitative findings (thinking behavior,
   tool calling, LM Studio compatibility, licensing).
7. Report: `llmbench report`; summarize the new model against the existing
   table, including context ceiling and caveats.

For a batch, use `llmbench suite --all` (skips models that cannot fit; resumes
unless `--force`).

## Models needing a custom backend

Ternary-Bonsai PQ2_0/PTQ1_0 requires Prism ML's fork; the `prism` profile is
auto-selected by filename. Stock llama.cpp and LM Studio cannot load it. If a
new quantization needs another fork, add a `[backends.<name>]` profile with
`binary`/`binary_glob`, `ld_library_path`, `flags`, and `match`, rather than
bypassing the tool.

## Serving Bonsai

`bonsai` (repo `scripts/bonsai`) starts the Prism fork server with 64k q8_0 KV,
reasoning effort medium, Bonsai's recommended sampling defaults, and the
built-in web UI at <http://127.0.0.1:8080>. Options: `--ctx`, `--kv`, `--port`,
`--reasoning`, `--no-open`. Ctrl+C stops it. Do not use LM Studio for Bonsai.

## Fairness rules

- Keep `-ngl 99 -fa on -b 2048 -ub 512 --parallel 1` identical across models.
- Speed numbers are server-reported `/completion` timings with `ignore_eos`.
- Report medians, mention min-max when relevant, and flag partial offload or
  high desktop VRAM.
- Do not compare runs made with different flags, backends, or KV settings
  without saying so.
