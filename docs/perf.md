# Perf: V runner vs Python on identical hardware

Reproduce: `python scripts/perf_baseline.py --kernels [--api --v-exe .\server.exe]`
(V kernel probe is a throwaway copy of `src/infer` hot loops; gated
versions live in `bench_test.v`.)

## Machine

11th Gen Intel i5-11400 (6C/12T), 30.8 GB RAM, Windows. Python 3.14.6,
NumPy 2.5.4 (OpenBLAS/MKL BLAS). V 0.5.2, `-prod` (`-O3`, asserts off).
Same client (urllib, fresh TCP connection per request — both servers send
`Connection: close`), same payload shape, back-to-back runs.

## Scope limits (read first)

* No weights on disk and the V forward pass is still pseudo-logits
  (phase 1b), so a **same-model tokens/sec shootout is meaningless today**
  and was deliberately not run. Anyone quoting one is comparing a real
  transformer against a stub.
* This compares (1) **kernel throughput** on identical shapes and
  (2) **HTTP serving overhead** on identical stub payloads.

## Kernels

| bench | V (`-prod`) | Python/NumPy | verdict |
|---|---|---|---|
| matvec 256×512 ×20 | 439 µs | 179 µs | NumPy **2.5×** (BLAS) |
| matvec 2048×6144 ×3 (FFN scale) | 72 ms | 5 ms | NumPy **14×** (multithreaded BLAS) |
| sample 64-logit ×20k | 38 ms | 707 ms | **V 18.6×** (NumPy per-call overhead on tiny arrays) |
| sample 151936-logit ×20 (vocab scale) | 141 ms | 31 ms | NumPy **4.5×** (vectorized exp/partition vs scalar loop) |

## Serving overhead (`POST /v1/chat/completions`, n=200, warmed up)

| server | mean | min | p50 | p99 |
|---|---|---|---|---|
| V `server.exe -prod` | 4.17 ms | 1.56 ms | 2.24 ms | 34.18 ms |
| Python `http.server` stub | 1.83 ms | 1.23 ms | 1.62 ms | 3.67 ms |

Note the V side does more per request (JSON decode + template + the
64-step generate loop + escape + framing) while the Python stub returns a
precomputed payload — so this flatters Python and still lands both in
single-digit ms. Unwarmed runs were noisier (medians flip-flopped between
runs: V p50 13.66→2.05 ms, Python p50 1.75→11.49 ms across runs), which is
why the reported numbers use 20 warmup requests.

## What it means

1. **Serving overhead is a wash.** ~2 ms vs ~1.6 ms median on stub
   payloads; both vanish next to real token-generation time (seconds).
   The V p99 outlier (~34 ms, likely spawn-per-request + GC pause) is the
   thing to watch — a bounded worker pool / keep-alive would fix it, at
   the cost of kernel-port complexity (`kernel/mini.md` tradeoff is
   deliberate).
2. **Matvec is the gap.** 14× at FFN scale is BLAS (threads + SIMD) vs a
   scalar loop. Phase-1b forward needs either a SIMD kernel (AVX2 via `C`
   intrinsics, bare-metal compatible) or a BLAS link on the host path.
   `q8_0_dot_row` already avoids half the traffic; the multiply itself is
   next.
3. **Sampler is fine.** V beats NumPy 18× on small logits and trails 4.5×
   at vocab scale — acceptable for a per-token tail operation, and the
   fused softmax + zero-alloc path keeps it GC-clean.

## Token throughput (sampling + serving path)

End-to-end model tok/s is not measurable yet (no weights on disk, V
forward still pseudo-logits — see scope limits). What *is* measured is
the throughput of every stage the V runner owns today:

| stage | V | Python/NumPy | how measured |
|---|---|---|---|
| 64-logit sample | **526k tok/s** (38 ms / 20k iters) | 28k tok/s (707 ms) | `sample_into` vs equivalent NumPy sampler, warmed up |
| vocab-scale sample (151936) | ~140 tok/s (141 ms / 20 iters) | ~640 tok/s (31 ms) | same, incl. full softmax + top-k/p |
| stub chat round trip (64 tokens/response) | ~16–30k tok/s served (~2–4 ms/resp) | ~40k tok/s served (~1.6 ms/resp) | `POST /v1/chat/completions`, n=200, 20 warmups |

The 64-logit rate is the V sampler ceiling (~1.9 µs/token); the
vocab-scale rate (~7 ms/token) is the softmax+top-k floor the real
forward will add to every token before matmuls even start.

## Per-token resource budget, Qwen3-1.7B (analytic, exact math)

Decode is memory-bound: each token re-reads all weights once.

| resource | Q8_0 (1.8 GB) | Q4_K_M (1.1 GB) |
|---|---|---|
| weight traffic / token | 1.8 GB | 1.1 GB |
| floor on this box (~35 GB/s DDR4) | ~50 ms → **~20 tok/s ceiling** | ~30 ms → **~33 tok/s ceiling** |
| KV-cache / token (28 layers × 2 × 8 KV heads × 128 dim) | 229 KB f32 (115 KB f16) | same |
| KV-cache @ 4k ctx | ~917 MB f32 | ~917 MB f32 |
| FLOPs / token (decode) | ~3.5 GFLOP | ~3.5 GFLOP |

Consequences already baked into the design: the 4k ctx cap
(`kernel/mini.md`) — full 32k ctx needs ~7 GB KV alone; the fused
`q8_0_dot_row` (never materializes dequantized rows); the 4 GB guest
minimum (1.8 GB weights + ~1 GB KV + page cache headroom).

## How to reproduce

```sh
python scripts/perf_baseline.py --kernels
v -prod -o server.exe src
python scripts/perf_baseline.py --api --v-exe .\server.exe
```

Kernel probe sources: throwaway copy of `src/infer` hot loops (gated
copies in `bench_test.v`); NumPy side in `scripts/perf_baseline.py`.
Same shapes, equivalent work, warmed up, back-to-back on the machine in
`docs/perf.md` header. Re-run on your hardware and replace the tables —
do not quote these numbers for other machines.

## Missing for end-to-end tok/s

1. Weights: `Qwen3-1.7B-Q8_0.gguf` (~1.8 GB) — same file both sides.
2. Python side: `llama-cpp-python`, 6 threads → prefill tok/s, decode
   tok/s, peak RSS.
3. V side: phase-1b forward (embed gather → 28× GQA + QK-norm + RoPE +
   SwiGLU → vocab head, Q8 rows streaming, real BPE merges), gated by
   text parity with llama.cpp output.
