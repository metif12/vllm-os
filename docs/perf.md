# Perf: V runner vs Python on identical hardware

Reproduce: `python scripts/perf_baseline.py --kernels [--api --v-exe .\server.exe]`
(V kernel probe is `tools/bench_kernels.v`, built two ways below; gated
copies live in `src/infer/bench_test.v`.)

## Machine

AMD Ryzen 7 5800HS (8C/16T), 15.4 GB RAM, Windows. Python 3.13.12,
NumPy 2.5.4. V 0.5.2, gcc 15.2.0. Same client (urllib, fresh TCP
connection per request — both servers send `Connection: close`), same
payload shape, back-to-back runs. Kernel rows measured 2026-11-14;
full-decode rows (`tools/forward_bench.v`) measured 2026-11-15, same box.

Previous table (11th Gen i5-11400, tcc-era V, selection-sort sampler)
is superseded — different machine AND different code. The deltas that
are code, not silicon: gcc instead of tcc (~7x on wide matvec), heap
top-k in the sampler (~7x on vocab scale), C AVX2 kernels via
function-target attributes, opt-in fast-softmax sampler.

## Scope limits (read first)

* Weights on disk since phase 1b: **Qwen3-0.6B-Q4_0** (382 MB) + tokenizer.
  The V forward pass is real (GQA + QK-norm + split-half RoPE decode,
  fused Q4_0/Q4_1/Q6_K matvecs, real BPE) — same-model tok/s below is
  quoted for that model on the machine in the header only. The old
  pseudo-logit comparison is retired; anyone quoting these numbers for
  other hardware is comparing against a different wall.
* This compares (1) **kernel throughput** on identical shapes,
  (2) **HTTP serving overhead** on identical stub payloads, and
  (3) **full-decode throughput** (section below) on the 0.6B file.

## Kernels

`v-rel` = `v -prod -cc gcc` (portable flags; AVX2 via function
attributes + runtime dispatch). `v-lto` adds redundant
`-cflags "-O3 -fno-strict-aliasing -flto"` — kept in the table as
evidence that extra flags don't move the needle (the profile was
retired; plain release is optimal). Each cell is the
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
| q4_0 dot 2048 blocks x20, **fused C AVX2** (split-half) | **~183 us** | — | ~4956 us materialize-then-dot (same bytes) | **fused 27x** (missing 32-f32 store + SIMD; the strided materialize reference has worse locality than the old adjacent one, so read this as fused-vs-naive, not SIMD-alone) |
| q4_0 matvec 3072x1024 (FFN gate scale), pure V | ~4 ms | — | — | baseline (fused, split-half, no materialize) |
| q4_0 matvec 3072x1024, **C AVX2** | **<1 ms** (timer floor) | — | — | **V 4x+** (epi64 nibble-unpack + FMA; real-data maxdiff 0.0) |
| q4_1 matvec 1024x3072 (FFN down scale), pure V | ~4 ms (3.6-4.3) | — | — | baseline (fused split-half + xsum*min, no materialize) |
| q4_1 matvec 1024x3072, **C AVX2** | **~0.7 ms** (14 ms/20) | — | — | **V ~5-6x** (unsigned-nibble unpack + in-register xsum; real-data maxdiff 0.0) |
| q6_k matvec 151936x1024 (vocab head), pure V | ~230 ms | — | — | baseline (warm box; cold-start 2.2 s is page-cache) |
| q6_k matvec 151936x1024, **C AVX2** | **~24 ms** | — | — | **V 9.5x** (16-lane unpack, per-group scale broadcast; maxdiff ~4e-6) |

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
6. **Q4_0 AVX2 matvec** (`vllm_q4_0_matvec_f32`): FFN gate 3072x1024
   ~4 ms pure-V -> sub-ms (timer floor), real-data maxdiff 0.0.
   Decode 870 ms -> ~340 ms (~2.5x) on the 0.6B file.
7. **Q6_K AVX2 matvec** (`vllm_q6_k_matvec_f32`): vocab 151936x1024
   ~230 ms pure-V -> ~24 ms (~9.5x), maxdiff ~4e-6. Decode
   340 ms -> 106 ms serial (~3.2x this stage; ~8x cumulative).
8. **Threaded vocab head** (`q6_k_matvec_parallel`): row-block split
   across `parallel_threads()` workers via `q6_k_matvec_c_range`.
   Serial 106-118 ms -> threaded 89-99 ms (~15% on full decode; the
   head is ~25% of the token). `VLLM_THREADS=1` / freestanding pins
   the serial kernel — the bare-metal guest stays single-threaded.
9. **Q4_1 AVX2 matvec** (`vllm_q4_1_matvec_f32`): FFN down 1024x3072
   ~4 ms pure-V -> ~0.7 ms (~5-6x), real-data maxdiff 0.0. The 3/28
   Q4_1 down-projections stop being the slow layers: serial decode
   ~118 ms -> ~100 ms, threaded ~99 ms -> ~80-90 ms (~10x cumulative).
10. **Layer fork-join** (`q4_0_matvec_2way`/`3way`): q/k/v and gate/up
    run as concurrent independent streams (one spawn per member, not
    per row-block — row-splitting these 1024-3072-row shapes regresses,
    measured). Threaded decode ~85 ms -> ~78 ms; serial path unchanged
    (same kernels, same order — threaded vs serial-pin is bit-identical).
    `VLLM_THREADS=1` / freestanding pin the serial joins — the guest
    stays single-threaded.

The wide-matvec story is the memory wall, moved twice: first SIMD
minimized per-byte cost (~11 ms -> ~8 ms single-threaded), then threads
multiplied available bandwidth up to the socket wall (~8 ms -> ~4 ms at
~37 GB/s effective). NumPy's multithreaded BLAS sits at ~7 ms on the
same wall. Remaining traffic wins come from quantization (less traffic
per token via the fused `q8_0_dot_row` path), not from kernel tuning.
Threading is spawn-per-call today (no pool): fine at 2048+ rows where
tens of microseconds vanish next to milliseconds; a persistent pool is
documented follow-up for the phase-1b forward, not needed for the win.

## Full-decode throughput (Qwen3-0.6B-Q4_0, same box)

`tools/forward_bench.v` (`-prod`, warm, back-to-back, in-tree reproduce —
see below). Best-of-N in ms/token; ranges are the samples, not CI gates.

| path | prefill 2 tok | decode/token | vs warm pure-V |
|---|---|---|---|
| pure-V (`VLLM_NO_SIMD=1`, serial) | ~1.8 s | ~920 ms (922-1032) | baseline |
| C AVX2, serial (`VLLM_THREADS=1`) | ~0.20-0.29 s | ~100 ms (98-108, best 98) | **~9x** |
| C AVX2, threaded (default 4) | ~0.15-0.23 s | **~78 ms** (78-79, best 78) | **~11x** |

Scratch prof harness (same box, same day, back-to-back, not in-tree):
q4_0 row-split of 1024-3072-row shapes NEVER wins (spawn dominates —
gate 3072x1024 serial 0.6 ms vs 2/4-way slower); fork-join over the
independent q/k/v + gate/up matvecs wins 0-20% per rep on the 7-matvec
layer set (interleaved reps: 56/57, 66/52, 59/54 ms serial/forkjoin).
In-forward that lands as threaded 80-96 ms -> 78-79 ms stable, serial
unchanged. Quote **~78-80 ms threaded, ~100 ms serial, ~920 ms pure-V**.
Cumulative stage-1+2+3+4+5: **870 ms → ~78 ms ≈ 11x** (warm pure-V →
threaded AVX2 + fork-join).

Correctness held at every step (same box, paired): greedy 8-step ids
identical (1096, 374, 264, 4285, 3110, 315, 1246, 311), re-verified
post-Q4_1 AVX2 vs pure-V (logit sums rel Δ ~5e-8); fork-join is
bit-identical by construction (same kernels, disjoint outs) — threaded
vs `VLLM_THREADS=1` logit sums match to the unit, plus bit-exact
2-way/3-way unit tests. PPL 8828617 vs 8828611 (Δ6 micro-units),
vocab maxdiff ~4.3e-6, gate maxdiff 0.0, Q4_1 down maxdiff 0.0 (real
weights), musl serial-pin guest bit-identical (`MUSL-SERIAL-IDENTICAL`).
Agreement tests green non-prod + `-prod`; `vet` + `fmt -verify` clean.

For context, llama.cpp on the same file/box does ~63 tok/s (~16 ms/token)
— about 5x faster than the current V forward. Expected: llama blocks,
packs, and reuses across layers in ways this forward does not yet. The
remaining levers are Q4_0 threading / layer fork-join (profile-gated),
then host-side profiling against that 16 ms line.

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

1. **Kernels: ahead, every row.** V wins all 8 FP32 rows: 1.3x on small
   matvec, 1.75x on FFN-scale matvec (threads + AVX2), 28x / 1.3x /
   5.5x across sampler scales. Quantized rows are V-vs-V (no NumPy
   equivalent): Q4_0 AVX2 4x+ on gate scale, Q4_1 AVX2 ~5-6x on down
   scale, Q6_K AVX2 9.5x on vocab scale. The old "matvec is the gap"
   verdict is closed for FP32; current decode work is quantized-matvec
   + threading shaped.
2. **Serving overhead is a wash** (unchanged verdict, new numbers).
   Both medians ~1-1.5 ms on quiet runs, both p99 ~25 ms. The
   spawn-per-request + GC-pause p99 theory from the old table did not
   survive: Python's `http.server` shows the same tail.
3. **Sampler is done.** Exact beats NumPy 1.4x at vocab scale and
   28x on small logits; fast beats it 5.5x. Real-logit validation
   (phase 1b gate) passed without sampler changes — no further work
   planned here.
4. **Quantized dots/matvecs are fused.** Q4_0/Q4_1 rows never
   materialize: fused `q4_0_dot_row` beats dequant-then-dot, and the
   AVX2 matvecs (`q4_0_matvec_c`, `q4_1_matvec_c`, `q6_k_matvec_c`)
   keep that fusion in SIMD form — gate <1 ms, down ~0.7 ms, vocab
   ~24 ms. This is the traffic shape the forward pays per row; the win
   is the missing store + unpack-in-register, not just width.
5. **Attention lost its per-row tax.** `attention_step` indexes the KV
   arenas flat instead of `k.row(i)!` per row: identical checksums,
   4 ms -> ~3 ms at n=1024 x20 and 4 ms -> ~2-3 ms at n=4096 x5
   (dim=128). At 4k ctx x 28 layers x 8 KV heads the per-call saving
   multiplies across 224 calls/token.
6. **Thread partitions are balanced.** `parallel.v` (and the bench
   harness) spread `rows % n` remainder rows so blocks differ by at
   most one — same review fix as VSL `gemv_n`. Gated by a
   rows=2501/VLLM_THREADS=8 remainder test in `parallel_test.v`
   (FP32 + Q6_K paths).
7. **Full decode is measured, not modeled.** `tools/forward_bench.v`
   gives ~78-80 ms/token threaded, ~100 ms serial, ~920 ms pure-V
   on Qwen3-0.6B-Q4_0 (same box). Greedy ids identical to the
   pre-kernel forward, fork-join bit-identical to serial, PPL within
   micro-units, musl guest bit-identical. Against llama.cpp's
   ~16 ms/token on the same file, the gap is ~5x — the documented
   next levers, not a measurement artifact.

## External baselines: llama.cpp on real small models (same box)

`llama-b12376-bin-win-cpu-x64`, `-t 8`, `-p 128 -n 64`, back-to-back
with the rows above (box under load — treat absolutes as conservative,
ratios as the signal):

| model | size | PP tok/s | TG tok/s |
|---|---|---|---|
| Qwen3-0.6B Q4_0 (unsloth) | 359 MB | ~389 | ~63 |
| Llama-3.2-1B Q4_0 (bartowski) | 730 MB | ~219 | ~33 |

Both files parse with our `load_gguf_table` (310 tensors/404 MB table
for Qwen, 147/812 MB for Llama) after two robustness fixes this round:
KV arrays over 4096 entries are walked-not-retained (Qwen3 tokenizers
carry 152k strings; the boot path only needs arch KV), and tensor byte
sizes use exact block math for Q4_0/Q4_1/Q5/Q8_0. Analytic decode
ceilings from weight traffic (~0.36 GB/token Qwen, ~0.73 GB/token
Llama at ~20 GB/s effective) land at ~55 and ~27 tok/s — llama.cpp sits
at or above those lines. End-to-end V tok/s is now quotable (phase 1b
landed): **~13 tok/s** on Qwen3-0.6B-Q4_0 (78 ms/token threaded, same
box) vs llama.cpp's ~63 tok/s on the same file — a ~5x gap that sizes
the remaining kernel/threading work. Known gap found while probing:
`config_from_kv` only understands `qwen3.*` keys, so the Llama arch line
above prints Qwen defaults (hidden right, layers/intermediate wrong) —
generalizing it rides with the tokenizer-from-GGUF follow-up, not this
round.

## Flag hygiene (measured, non-obvious)

* Custom `-cflags` APPEND to V's defaults (verified: `-dump-c-flags`
  shows the same `-O3 -flto` plan with and without extras, and
  `-prod` / `-prod -cflags "-O3"` / `-prod -cflags "-O3
  -fno-strict-aliasing"` build bit-identical-speed binaries on the
  bench). An earlier "REPLACE" theory in these docs was wrong —
  re-measured 2026-11-14 and corrected. Plain `-prod -cc gcc` is
  already optimal; extra flags are redundant.
* `-march=native` (and plain `-mavx2`) pessimize V 0.5.2's generated
  4-accumulator matvec ~2.4x (8-9 ms -> 21 ms, identical checksums;
  minimal repro below) while a hand-written C kernel with the same
  loop shape keeps full AVX2 speed (145 us). Root cause, from the
  generated C + asm: V emits per-access `*((float*)(arr.data) + i64…)`
  addressing that gcc's SLP pass turns into a `vpermd`/`vblendps`
  shuffle swamp with a scalar `vaddss` horizontal tail, where the
  plain-C `const float*` version gets clean 8-wide FMA. `-mavx2
  -fno-tree-vectorize` recovers most of it, confirming the vectorizer
  (filed upstream as a vlang/v issue with repro + asm).
* SIMD therefore comes from `__attribute__((target("avx2,fma")))` +
  runtime dispatch — portable flags, full speed where AVX2 exists.
  Never `-march=native` in committed scripts (also an ISO
  portability rule).

Minimal upstream repro (save as `march_repro.v`):

```v
module main

import time

@[direct_array_access]
fn matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	for r in 0 .. rows {
		mut s0 := f32(0)
		mut s1 := f32(0)
		mut s2 := f32(0)
		mut s3 := f32(0)
		base := r * cols
		mut c := 0
		for c + 4 <= cols {
			s0 += w[base + c] * x[c]
			s1 += w[base + c + 1] * x[c + 1]
			s2 += w[base + c + 2] * x[c + 2]
			s3 += w[base + c + 3] * x[c + 3]
			c += 4
		}
		for c < cols {
			s0 += w[base + c] * x[c]
			c++
		}
		out[r] = s0 + s1 + s2 + s3
	}
}

fn main() {
	rows := 2048
	cols := 6144
	mut w := []f32{len: rows * cols}
	mut x := []f32{len: cols}
	for i in 0 .. w.len { w[i] = f32(i % 13) * 0.125 }
	for i in 0 .. x.len { x[i] = f32(i % 7) * 0.25 }
	mut out := []f32{len: rows}
	matvec(w, x, rows, cols, mut out) // warmup
	sw := time.new_stopwatch()
	for _ in 0 .. 3 {
		matvec(w, x, rows, cols, mut out)
	}
	println('matvec 2048x6144 x3: ${sw.elapsed().milliseconds()}ms checksum=${out[0] + out[rows-1]}')
}
```

```
v -prod -cc gcc -o repro_def march_repro.v && ./repro_def        # ~8-9ms
v -prod -cc gcc -cflags "-mavx2" -o repro_avx2 march_repro.v && ./repro_avx2  # ~21ms
v -prod -cc gcc -cflags "-mavx2 -fno-tree-vectorize" -o repro_novec march_repro.v  # ~11ms
```
(checksums identical: `6906.5`; Ryzen 7 5800HS, gcc 15.2, V 0.5.2)

## Token throughput (real forward + sampling + serving path)

End-to-end model tok/s is measured since phase 1b (weights on disk, real
forward — see scope limits). The stub rows stay for the serving-path
record; the forward row is the one to quote for model speed:

| stage | V | Python/NumPy | how measured |
|---|---|---|---|
| full decode, Qwen3-0.6B-Q4_0 (threaded) | **~13 tok/s** (78 ms best, 78-79 range) | llama.cpp ~63 tok/s same file/box | `tools/forward_bench.v`, `-prod`, warm, back-to-back |
| full decode, same file (serial AVX2) | ~10 tok/s (101 ms best, 98-108) | — | `VLLM_THREADS=1`, same probe |
| full decode, same file (pure-V) | ~1.1 tok/s (922 ms best, 922-1032) | — | `VLLM_NO_SIMD=1`, same probe |
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
python scripts/bench.py --force            # V bench + NumPy, fingerprinted, CSV/JSON in bench/results_*
python scripts/bench.py --force --llama-bench /path/to/llama-bench --pp 128 -n 64 -t 8
python scripts/bench.py --help             # reps/cooldown/models/threads
```

`bench.py` refuses on a loaded box unless `--force`, sleeps `--cooldown`
between sections, and records CPU/RAM/V/gcc/commit with every run. The
manual equivalents still work:

```sh
v -prod -cc gcc -o /tmp/bench_rel tools/bench_kernels.v && /tmp/bench_rel
python scripts/perf_baseline.py --kernels
v -prod -cc gcc -o server.exe src
python scripts/perf_baseline.py --api --v-exe .\server.exe
# full-decode probe (needs a GGUF file; Qwen3-0.6B-Q4_0 measured here):
v -prod -cc gcc -path "src|@vlib|@vmodules" -o /tmp/fwdbench tools/forward_bench.v
/tmp/fwdbench /path/to/Qwen3-0.6B-Q4_0.gguf 5
VLLM_THREADS=1 /tmp/fwdbench /path/to/Qwen3-0.6B-Q4_0.gguf 3
VLLM_NO_SIMD=1 VLLM_THREADS=1 /tmp/fwdbench /path/to/Qwen3-0.6B-Q4_0.gguf 2
```

Kernel probe sources: `tools/bench_kernels.v` embeds verbatim copies of
the pure-V kernels (SYNC-marked to `src/infer`) and links the REAL
`src/infer/vllm_kernels.c` — no copy, always in sync. Full-decode probe:
`tools/forward_bench.v` imports the REAL `infer` + `formats` modules
(`-path "src|@vlib|@vmodules"`, print/println only for the release-V
`-prod` tiny-binary quirk) and times prefill + N decodes on a real GGUF
file — no copies, always the shipped forward. NumPy side in
`scripts/perf_baseline.py`. Same shapes, equivalent work, warmed up,
back-to-back on the machine in `docs/perf.md` header. Re-run on your
hardware and replace the tables — do not quote these numbers for other
machines.

## Missing for end-to-end tok/s parity (not for quoting V numbers)

V decode is quotable now (row above). What is missing is the
same-model, same-box Python-side pair and the remaining V levers:

1. Python side: `llama-cpp-python` (or `llama-bench` TG line) on the
   exact `Qwen3-0.6B-Q4_0.gguf` file measured above — the llama.cpp
   binary row (~63 tok/s) is the stand-in until then.
2. V levers: blocking/packing the weight layout (llama-style reuse
   across layers), attention-path work, then host-side profiling
   against the 16 ms/token llama line. Q4_1 AVX2 and q/k/v + gate/up
   fork-join both landed (this round + last).
3. Serve path: real-model `POST /v1/chat/completions` tok/s (stub rows
   only cover HTTP overhead, not the forward).
