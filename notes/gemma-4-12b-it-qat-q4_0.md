# Gemma 4 12B IT QAT (Q4_0)

- Multimodal (vision) model; the only current model with image input.
- ~66.3 t/s generation and ~1,319 t/s prefill in the manual harness.
- Sliding-window attention on most layers keeps long-context VRAM low. A manual
  test loaded the full 256k native context with q8_0 KV at ~10.9 GB total.
- QAT quantization keeps quality close to the larger Gemma builds at 12B size.
- Use when images or very long context matter; Ornith/Qwen 9B are faster for
  plain text.
