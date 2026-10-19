# Ternary Bonsai 2 27B (PQ2_0)

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
