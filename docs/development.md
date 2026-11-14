# Development

Host-only dev dependency (parity tests + tools, never the server):

```sh
v install vtl vsl
```

## Test suite

```sh
v -cc gcc test src                       # dev: pure V (whole tree; per-module dirs shadow `cfg` vs vlib)
v -prod -cc gcc test src                  # prod: C layer live
```

tcc is never used for tests either.

| file | covers |
|---|---|
| `src/infer/infer_test.v` | greedy/softmax/seed-determinism/RMSNorm/Q8 scale/template/bad-magic/KV override, presets, min-p, tokenizer dispatch |
| `src/infer/bench_test.v` | perf gates: 20k-iteration `sample_into` floor, blocked-vs-naive matvec equivalence + speed |
| `src/infer/attention_test.v` | KV-cache append/overflow/row/reset, `attention_step` hand values + error gates |
| `src/infer/backend_test.v` | registry contents (4 backends), explicit/auto selection, backend agreement |
| `src/infer/simd_test.v` | C-vs-V agreement (matvec/Q8-dot/RMSNorm/range-block), heap top-k winners, fast-vs-exact sampler |
| `src/infer/parallel_test.v` | threaded-vs-serial agreement (incl. wide + non-divisible rows), range-block parity |
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
