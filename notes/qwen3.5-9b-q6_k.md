# Qwen3.5 9B (Q6_K)

- Strong general-purpose 9B: ~76.2 t/s generation, ~1,215 t/s prefill in the
  manual harness.
- Good default for everyday chat and general tasks; a little slower to prefill
  than Ornith but similarly fast at generation.
- Fully GPU-resident at 8k (~7.7 GB total). 128k with q8_0 KV previously measured
  at ~10.2 GB total; 192k does not leave enough headroom.
