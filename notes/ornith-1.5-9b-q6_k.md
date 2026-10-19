# Ornith 1.5 9B (Q6_K)

- Historically the fastest model on this machine: ~77.4 t/s generation and
  ~3,353 t/s prefill in the manual harness.
- Best pure prefill performance of the current lineup; well suited to agent
  loops and repeated long-prompt work.
- Strong software-engineering benchmark results for its size (best of the 9B
  class in the manual comparison).
- Fully GPU-resident at 8k (about 7.6 GB total). 128k with q8_0 KV previously
  measured at ~10.1 GB total; 192k is too tight.
