# Perf: V runner vs Python on identical hardware

Reproduce: `python scripts/perf_baseline.py --kernels [--api --v-exe .\server.exe]`
(V kernel probe is `tools/bench_kernels.v`, built two ways below; gated
copies live in `src/infer/bench_test.v`.)

## Machine

AMD Ryzen 7 5800HS (8C/16T), 15.4 GB RAM, Windows. Python 3.13.12,
NumPy 2.5.4. V 0.5.2, gcc 15.2.0. Same client (urllib, fresh TCP
connection per request — both servers send `Connection: close`), same
payload shape, back-to-back runs. Measured 2026-11-14.

Previous table (11th Gen i5-11400, tcc-era V, selection-sort sampler)
is superseded — different machine AND different code. The deltas that
are code, not silicon: gcc instead of tcc (~7x on wide matvec), heap
top-k in the sampler (~7x on vocab scale), C AVX2 kernels via
function-target attributes, opt-in fast-softmax sampler.

## Scope limits (read first)

* No weights on disk and the V forward pass is still pseudo-logits
  (phase 1b), so a **same-model tokens/sec shootout is meaningless today**
  and was deliberately not run. Anyone quoting one is comparing a real
  transformer against a stub.
* This compares (1) **kernel throughput** on identical shapes and
  (2) **HTTP serving overhead** on identical stub payloads.

## Kernels

`v-rel` = `v -prod -cc gcc` (portable flags; AVX2 via function
attributes + runtime dispatch). `v-lto` adds
`-cflags "-O3 -fno-strict-aliasing -flto"`. Each cell is the
median-ish of back-to-back runs (ranges in text). Machine section above
is the only hardware these numbers may be quoted for.

| bench | v-rel | v-lto | Python/NumPy | verdict |
|---|---|---|---|---|
| matvec 256x512 x20, pure V | ~425 us | ~429 us | ~190 us | NumPy 2.3x (BLAS) |
| matvec 256x512 x20, **C AVX2** | **~145 us** | **~144 us** | ~190 us | **V 1.3x** |
| matvec 2048x6144 x3 (FFN scale), pure V | ~9 ms | ~11 ms | ~7 ms | NumPy 1.3x |
| matvec 2048x6144 x3, **C AVX2** | ~6 ms (5-8) | ~8 ms (7-14) | ~7 ms | **V ~1.2x** (variance overlaps; threads win clean) |
| matvec 2048x6144 x3, **threads + AVX2** | **~4 ms** (`cpu-parallel`, 2-4 threads) | — (same code) | ~7 ms | **V 1.75x** |
| sample 64-logit x20k, exact | ~30 ms | ~30 ms | ~830 ms | **V 28x** (NumPy per-call overhead on tiny arrays) |
| sample 151936-logit x20, exact (heap top-k) | ~19 ms | ~19 ms | ~24 ms | **V 1.3x** |
| sample 151936-logit x20, **fast** (approx exp) | **~4 ms** | **~5 ms** | ~24 ms | **V 5.5x** |

How the wins happened, in order:

1. **Compiler (tcc -> gcc -prod):** wide matvec 72 ms -> ~10 ms
   before any new code. tcc codegen was the single biggest tax.
2. **Heap top-k** (`sample_from_probs`): vocab exact 141 ms -> ~20 ms.
   Selection sort's O(40 x 152k) comparisons were the bottleneck, not exp.
3. **C AVX2 matvec** (`vllm_matvec_f32`): small 440 us -> ~150 us,
   wide ~11 ms -> ~8 ms. Single-threaded, portable binary.
4. **Fast softmax** (`sample_into_fast`, branchless approx exp):
   vocab 20 ms -> 5 ms. Opt-in only — the server and all tests use exact.
5. **Threads + AVX2** (`matvec_parallel`, `cpu-parallel` backend):
   wide 5-8 ms -> ~4 ms (2-4 threads). One thread cannot saturate this
   box's memory controllers (~24 GB/s); two threads reach ~37 GB/s
   and the wall moves out to ~4 ms. 8 threads add contention without
   more bytes (5-6 ms). Small shapes stay serial — spawn costs more
   than threading saves below `min_rows_parallel` (2048).

The wide-matvec story is the memory wall, moved twice: first SIMD
minimized per-byte cost (~11 ms -> ~8 ms single-threaded), then threads
multiplied available bandwidth up to the socket wall (~8 ms -> ~4 ms at
~37 GB/s effective). NumPy's multithreaded BLAS sits at ~7 ms on the
same wall. Remaining traffic wins come from quantization (less traffic
per token via the fused `q8_0_dot_row` path), not from kernel tuning.
Threading is spawn-per-call today (no pool): fine at 2048+ rows where
tens of microseconds vanish next to milliseconds; a persistent pool is
documented follow-up for the phase-1b forward, not needed for the win.

## Serving overhead (`POST /v1/chat/completions`, n=200, warmed up)

| server | mean | min | p50 | p99 |
|---|---|---|---|---|
| V `server.exe` release (gcc -prod) | 5.62 ms | 1.03 ms | 1.44 ms | 28.30 ms |
| V `server.exe` lto | 5.15 ms | 0.96 ms | 1.35 ms | 25.26 ms |
| V `server.exe` minimal (no -prod, -Os) | 9.56 ms | 1.11 ms | 5.86 ms | 39.56 ms |
| Python `http.server` stub (3 runs) | 2.70 / 5.34 / 5.46 ms | 0.69 ms | 0.88 / 1.20 / 1.21 ms | 17.9 / 22.1 / 25.1 ms |

Both sides live in the same 1-6 ms band with ~25 ms p99 tails on
Windows loopback — the tail is the platform (fresh TCP connection per
request on both sides), not the runner. Note the Python side itself
swings 2.7 -> 5.5 ms mean between back-to-back runs: single-run
comparisons at this scale are noise. The V side does more per request
(JSON decode + template + the 64-step generate loop + escape + framing)
while the Python stub returns a precomputed payload.

Minimal costs ~4 ms p50 vs release: the price of the no-C,
asserts-on auditable binary. Release and lto are indistinguishable
here — serve release.

## What it means

1. **Kernels: ahead, every row.** V wins all 8 rows: 1.3x on small
   matvec, 1.75x on FFN-scale matvec (threads + AVX2), 28x / 1.3x /
   5.5x across sampler scales. The old "matvec is the gap" verdict is
   closed for FP32; remaining traffic wins come from quantization,
   not from kernel tuning.
2. **Serving overhead is a wash** (unchanged verdict, new numbers).
   Both medians ~1-1.5 ms on quiet runs, both p99 ~25 ms. The
   spawn-per-request + GC-pause p99 theory from the old table did not
   survive: Python's `http.server` shows the same tail.
3. **Sampler is done.** Exact beats NumPy 1.4x at vocab scale and
   28x on small logits; fast beats it 5.5x. No further work planned
   here before real-logit validation (phase 1b gate).

## Flag hygiene (measured, non-obvious)

* Custom `-cflags` REPLACE V's default C flag set. Bare `-O3`
  measured ~2x slower than plain `-prod` on V-generated code; the
  working recipe restates `-O3 -fno-strict-aliasing` (see
  `scripts/build.ps1` host-native).
* `-march=native` pessimize V 0.5.2's own codegen 3-5x (naive
  matvec 425 us -> ~1400 us, exact sampler 20 ms -> ~110 ms) while
  the hand-written C kernel kept its AVX2 speed. SIMD therefore
  comes from `__attribute__((target("avx2,fma")))` + runtime
  dispatch — portable flags, full speed where AVX2 exists. Never
  `-march=native` in committed scripts (also an ISO portability rule).

## Token throughput (sampling + serving path)

End-to-end model tok/s is not measurable yet (no weights on disk, V
forward still pseudo-logits — see scope limits). What *is* measured is
the throughput of every stage the V runner owns today:

| stage | V | Python/NumPy | how measured |
|---|---|---|---|
| 64-logit sample | **~650k tok/s** (31 ms / 20k iters) | 23k tok/s (869 ms) | `sample_into` vs equivalent NumPy sampler, warmed up |
| vocab-scale sample (151936), exact | ~1000 tok/s (20 ms / 20 iters) | ~730 tok/s (28 ms) | same, incl. full softmax + heap top-k/p |
| vocab-scale sample, fast | **~4000 tok/s** (5 ms / 20 iters) | ~730 tok/s | approx-exp + heap top-k/p |
| stub chat round trip (64 tokens/response) | ~16-30k tok/s served (~2-6 ms/resp) | ~40k tok/s served (~1 ms/resp, quiet) | `POST /v1/chat/completions`, n=200, 20 warmups |

## Per-token resource budget, Qwen3-1.7B (analytic, exact math)

Decode is memory-bound: each token re-reads all weights once.

| resource | Q8_0 (1.8 GB) | Q4_K_M (1.1 GB) |
|---|---|---|
| weight traffic / token | 1.8 GB | 1.1 GB |
| floor on ~20 GB/s effective DDR4 | ~90 ms -> **~11 tok/s ceiling** | ~55 ms -> **~18 tok/s ceiling** |
| KV-cache / token (28 layers x 2 x 8 KV heads x 128 dim) | 229 KB f32 (115 KB f16) | same |
| KV-cache @ 4k ctx | ~917 MB f32 | ~917 MB f32 |
| FLOPs / token (decode) | ~3.5 GFLOP | ~3.5 GFLOP |

Consequences already baked into the design: the 4k ctx cap
(`kernel/mini.md`) — full 32k ctx needs ~7 GB KV alone; the fused
`q8_0_dot_row` (never materializes dequantized rows); the 4 GB guest
minimum. Ceiling scales with measured, not theoretical, bandwidth —
re-measure the floor on the serving box, not this laptop.

## How to reproduce

```sh
v -prod -cc gcc -o /tmp/bench_rel tools/bench_kernels.v && /tmp/bench_rel
v -prod -cc gcc -cflags "-O3 -fno-strict-aliasing -flto" -o /tmp/bench_lto tools/bench_kernels.v && /tmp/bench_lto
python scripts/perf_baseline.py --kernels
v -prod -cc gcc -o server.exe src
python scripts/perf_baseline.py --api --v-exe .\server.exe
```

Kernel probe sources: `tools/bench_kernels.v` embeds verbatim copies of
the pure-V kernels (SYNC-marked to `src/infer`) and links the REAL
`src/infer/vllm_kernels.c` — no copy, always in sync. NumPy side in
`scripts/perf_baseline.py`. Same shapes, equivalent work, warmed up,
back-to-back on the machine in `docs/perf.md` header. Re-run on your
hardware and replace the tables — do not quote these numbers for other
machines.

## Missing for end-to-end tok/s

1. Weights: `Qwen3-1.7B-Q8_0.gguf` (~1.8 GB) — same file both sides.
2. Python side: `llama-cpp-python`, 6 threads -> prefill tok/s, decode
   tok/s, peak RSS.
3. V side: phase-1b forward (embed gather -> 28x GQA + QK-norm + RoPE +
   SwiGLU -> vocab head, Q8 rows streaming, real BPE merges), gated by
   text parity with llama.cpp output.
