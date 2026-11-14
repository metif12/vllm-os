# Development

Host-only dev dependencies (parity tests need all three; the server and
all kernels need none):

```sh
v install vtl vsl
```

## Toolchain

Pinned: **V 0.5.2 release** (`v version` prints a second hash after the
version — `7647ce1` is the release; anything else is a weekly and will
NOT build this repo). CI downloads the release; use it locally too.

Why the pin is load-bearing, not cosmetic:

- Weeklies removed the stdlib `json` module (use `json2` instead) and
  their `v fmt -w` auto-migrates `import json` plus call sites — even
  inventing `escape_unicode`/`time_as_unix` args (verified by probe).
  Release 0.5.2 has no `json2` module at all. There is no import that
  compiles on both, so the repo targets release: stdlib `json` only
  (`Config`, API shapes, tokenizer index).
- Release 0.5.2 rejects `...` array spread and `&?T` (both used to
  compile here); keep them out.
- No module-level `fn C.*` re-declarations (`cfile.v`): on release they
  shadow vlib's `builtin/cfns` and break vlib/os itself. Use vlib's
  declarations with vlib's own call shapes.
- Formatting authority is the **release binary ONLY**. Never run a
  weekly's `v fmt -w` on this tree.

## Test suite

```sh
v -cc gcc test src                       # dev: pure V (whole tree; per-module dirs shadow `cfg` vs vlib)
v -prod -cc gcc test src                  # prod: C layer live
```

`src/infer/vtl_parity_test.v` needs `v install vtl vsl` and is excluded
from the release-0.5.2 CI run (vtl master does not compile on 0.5.2) —
run it explicitly wherever your VTL checkout builds; CI runs the exact
file list in `.github/workflows/ci.yml` instead of the whole tree.

tcc is never used for tests either.

CI (`.github/workflows/ci.yml`) runs on every push/PR: `vet` + `fmt -verify`
on Ubuntu, the pinned file list on Ubuntu/gcc, the same list on Windows,
plus a `-prod` bench build under MSVC (C kernel portability — this lane
once caught a `__builtin_sqrt` the scalar fallback could not compile).
Perf numbers are never gated in CI (boxes vary); `scripts/bench.py`
owns them.

| file | covers |
|---|---|
| `src/infer/infer_test.v` | greedy/softmax/seed-determinism/RMSNorm/Q8 scale/template/bad-magic/KV override, presets, min-p, tokenizer dispatch |
| `src/infer/gguf_golden_test.v` | real-file tables: Qwen3-0.6B + Llama-3.2-1B tensor counts/bytes/types/key shapes; skips when `models/` absent (see `scripts/fetch_small_models.sh`) |
| `src/infer/bench_test.v` | perf gates: 20k-iteration `sample_into` floor, blocked-vs-naive matvec equivalence + speed |
| `src/infer/attention_test.v` | KV-cache append/overflow/row/reset, `attention_step` hand values + error gates |
| `src/infer/backend_test.v` | registry contents (4 backends), explicit/auto selection, backend agreement |
| `src/infer/simd_test.v` | C-vs-V agreement (matvec/Q8-dot/**Q4-fused-dot**/RMSNorm/range-block), heap top-k winners, fast-vs-exact sampler |
| `src/infer/parallel_test.v` | threaded-vs-serial agreement (incl. wide + **remainder rows**), range-block parity |
| `src/infer/tensor_view_test.v` | view validation, `-1` reshape, matvec parity, strided-layout rejection |
| `src/infer/vtl_parity_test.v` | VTL `la.matmul` and VSL `matrix_matrix_mul` agree with `matvec`; view/VTL layout parity |
| `src/formats/formats_test.v` | GGUF/SafeTensors/unknown/PyTorch detection, header parse, dtype table, registry resolve + sniff parity |
| `src/cfg/config_test.v` | defaults, partial-file fallback, BOM strip, rejection of bad values, simd/auto backends, env overrides, preset resolve |

| `src/api/routes_test.v` | route match/miss, health + models payloads, sampler merge rules, max_tokens clamp, 400/200 round trips |

Rules (house policy): every fix ships with a failing-first test where
practical — break the line, watch it go red, restore it. A silent green
proves nothing: confirm the new test names appear in output.
Run the whole tree (`v test src`): per-module dirs resolve `import cfg`
against vlib's unrelated `config` module instead of ours.

## Conventions (the parts that bite)

* Token loop is allocation-free: hoist scratch buffers out, pass `mut`,
  never `clone()` per token. `sample()` allocates (compat); `sample_into()`
  does not — the loop uses the latter. `sample_into_fast()` is opt-in
  (approx exp) and never the default.
* Hot kernels get `[direct_array_access]`; validate shapes at model load,
  never per token.
* `infer` = pure compute (imports `formats` only). No `net`, no `os` wall
  clock in the loop; `os` file access is load-time only and goes through
  `cfile.v` probes on the kernel port. (`simd_c.v` reads `VLLM_NO_SIMD`
  once per `has_avx2()` call — boot/config time, never in the loop.)
* No custom `-d` flags in `src/`: V errors on undeclared `$if` names,
  so feature switches use builtins (`$if prod`, `$if freestanding`)
  + runtime (`VLLM_NO_SIMD`, `backend` knob). Custom `-cflags` append
  to V's defaults — plain `-prod` is already optimal; never
  `-march=native` on V code (SLP shuffle swamp, see `docs/perf.md`).
* `v fmt -w src` before every commit (CRLF/LF churn is the formatter's,
  ignore it). `v vet` warnings about missing docs on `pub` items are
  real — document them.
* Keep `Gen`/`generate` signatures stable: phase 1b swaps pseudo-logits
  for the transformer forward without touching the server.

## Where to extend

* New weight format → implement `formats.ContainerHandler` (registry).
* New compute target → implement `Backend` in `src/infer/backend.v`,
  register in `list_backends()` + `select_backend()`, add an agreement
  test against `CpuNaiveBackend`.
* Removing the C layer → delete `src/infer/simd_c.v`,
  `src/infer/vllm_kernels.{h,c}`, and the `CpuSimdBackend` block in
  `backend.v`; everything degrades to pure V. (Normally unnecessary:
  non-prod builds already exclude it, `VLLM_NO_SIMD=1` disables it.)
* New endpoint → `types.v` shape + `handlers.v` fn + one `routes()` row +
  `routes_test.v` cases; document in `docs/api.md`. Keep it socket-level: no `veb`.
* New sampler preset → add to `preset_params()` in `src/infer/sampler.v`;
  `src/cfg` validation picks it up via `infer.sampler_presets()`.
* New tokenizer → implement `infer.Tokenizer` (`encode` + `name`);
  the generate path and server need no changes.
* New knob → `src/cfg/config.v` field + default + range check +
  `vllm.example.json` + `docs/configuration.md` row + CLI override in
  `src/main.v` if it deserves a flag (+ `VLLM_*` env in `apply_env`).
* New kernel/OS feature → `kernel/features.conf` flag, default OFF,
  wire to the phase-3 fork build. Fallback drivers are deleted, not flagged.
