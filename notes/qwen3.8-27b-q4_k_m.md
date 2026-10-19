# Qwen3.8 27B (Q4_K_M) - does not fit

- 16.8 GB weight file; cannot run fully on a 12 GB card.
- Historical partial-offload setup: 29/65 layers on GPU, ~3.4 t/s generation
  with the manual harness (LM Studio reached ~5 t/s with MTP).
- Excluded from the fully-GPU benchmark suite. Kept as a reference point and as
  an optional quality-first slow lane; not directly comparable to the models in
  the main table.
