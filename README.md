# llm-model-benchmarks

Reproducible speed, VRAM and context benchmarks for local GGUF models on a
12 GB RTX 5070, restricted to models that run **fully on the GPU** (no CPU
offload).

The harness drives a `llama-server`-compatible backend, measures prefill and
generation throughput with server-reported timings, maps the context-vs-VRAM
ceiling over KV cache types, and generates a Markdown report.

## Quick start

```bash
llmbench list                          # discovered models, size, native ctx, fit
llmbench bench ornith-1.5-9b-q6_k      # speed + VRAM at 8k
llmbench context ornith-1.5-9b-q6_k    # 32k f16 ... 128k q4_0 ladder
llmbench sanity ornith-1.5-9b-q6_k     # one chat prompt, catches broken output
llmbench suite --all                   # bench + context for every fitting model
llmbench report                        # results/report.md
```

`llmbench` is a symlink to `llmbench.py` in this repo. Python 3.11+ and
`nvidia-smi` are required; no third-party packages.

## Serving Bonsai (daily use)

`scripts/bonsai` is symlinked to `~/.local/bin/bonsai`:

```bash
bonsai                         # 64k q8_0, reasoning medium, opens web UI
bonsai --ctx 32768 --kv f16    # quality-first daily config
bonsai --ctx 131072 --kv q4_0  # long-document config
bonsai --port 8081 --no-open   # second instance / no browser
```

It resolves the Prism binary and LM Studio's CUDA runtime, checks the model and
port, applies Bonsai's recommended sampling defaults, and serves the built-in
web UI at `http://127.0.0.1:8080`. Ctrl+C stops it. Override paths with
`BONSAI_PRISM_DIR` / `BONSAI_MODEL` if the install or SSD mount moves.

## Use from OpenCode

With `bonsai` running, OpenCode can use it as an OpenAI-compatible provider:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "bonsai": {
      "name": "Bonsai (local)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "bonsai": {
          "name": "Ternary Bonsai 2 27B (PQ2_0)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 65536, "output": 8192 },
          "compatibility": { "reasoningField": "reasoning_content" }
        }
      }
    }
  }
}
```

Then select **Bonsai (local)** in `/models`, or run
`opencode run --model bonsai/bonsai "..."`. Tool calling and streamed
reasoning are verified.

## Install

```bash
chmod +x llmbench.py
mkdir -p ~/.local/bin
ln -sf "$PWD/llmbench.py" ~/.local/bin/llmbench

# optional: OpenCode skill
mkdir -p ~/.config/opencode/skills
ln -sfn "$PWD/skill" ~/.config/opencode/skills/llm-bench
```

## Commands

| Command | Purpose |
|---|---|
| `list` | Scan `model_roots`, parse GGUF metadata, estimate VRAM fit |
| `bench <model>` | Warm-up, 3 repetitions, prefill/generation medians, VRAM |
| `context <model>` | Load at each ladder rung, record OK/FAIL + VRAM |
| `sanity <model>` | One chat completion with `--jinja`; stores reasoning + answer |
| `suite --all` | Bench + context; skips oversized models, resumes by default |
| `report` | Build `results/report.md` from stored JSON |

Model arguments accept a slug (`ornith-1.5-9b-q6_k`), filename, substring, or
absolute path. `suite --force` redoes existing results; `--include-oversized`
tries models that do not fit.

## Configuration

All machine-specific settings live in `config.toml`:

- `general.model_roots` - directories scanned for `*.gguf` (mmproj excluded).
- `general.default_backend` - fallback backend.
- `speed` - measurement context (`8192`), KV type (`f16`), reps (`3`), generated
  tokens (`128`).
- `context.ladder` - context/KV rungs to try; rungs above the model's native
  context are skipped.
- `backends.*` - binary path or glob, `LD_LIBRARY_PATH`, shared flags, and
  optional filename `match` patterns for auto-selection.

Two backends are preconfigured:

- `lmstudio` - newest CUDA build under `~/.lmstudio/extensions/backends/`, using
  the CUDA runtime LM Studio ships.
- `prism` - Prism ML's llama.cpp fork, required by Ternary-Bonsai PQ2_0/PTQ1_0.
  Auto-selected for filenames matching `*Bonsai*`, `*PQ2_0*`, `*PTQ1_0*`.

### Adding a backend

```toml
[backends.custom]
binary = "~/path/to/llama-server"
ld_library_path = "~/path/to/cuda/runtime"   # optional
match = ["*MyQuant*"]                          # optional
flags = ["-ngl", "99", "-fa", "on", "-b", "2048", "-ub", "512",
         "--parallel", "1", "--no-webui"]
```

Keep `flags` identical across backends so results remain comparable.

## Methodology

- One slot, `-ngl 99`, `-fa on`, `-b 2048 -ub 512`, `--parallel 1`, no
  speculative decoding.
- **Warm-up is discarded**: one short generation plus one full-length prefill.
  Cold kernel/module init can otherwise halve the first prefill number.
- Speed: ~1.2k prompt tokens, exactly 128 generated tokens (`ignore_eos`),
  temperature 0, `cache_prompt: false`, 3 reps, median reported.
- VRAM: total `nvidia-smi` usage after load, including the desktop baseline.
- Context: server start success at each rung; failure classified as OOM /
  context-failed / missing-runtime / load-error.

## Fully-GPU policy

Models whose weights exceed available VRAM are out of scope. `list` estimates
the requirement from file size + ~900 MiB runtime overhead; `suite --all` skips
those unless `--include-oversized` is passed. This keeps the comparison honest:
no CPU-offloaded/slow-lane numbers mixed in.

## Output

```
results/
├── <slug>/
│   ├── speed.json      # medians, raw reps, VRAM, backend, environment
│   ├── context.json    # per-rung result + failure mode
│   └── sanity.json     # chat smoke test
├── logs/               # server logs (gitignored)
└── report.md           # generated Markdown
notes/<slug>.md         # qualitative notes, appended to the report
```

Add notes for things the harness cannot measure: thinking/tool-calling behavior,
LM Studio compatibility, licensing, quality impressions.

## Report

`llmbench report` emits environment, at-a-glance winners, a speed table, a
context matrix with failure modes, warnings, per-model notes, and the exact
methodology. Commit `results/**/*.json` and `results/report.md`; logs stay local.
