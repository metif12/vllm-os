# Development

## Test suite

```sh
v test src/infer src/formats src/config
```

| file | covers |
|---|---|
| `src/infer/infer_test.v` | greedy/softmax/seed-determinism/RMSNorm/Q8 scale/template/bad-magic/KV override |
| `src/infer/bench_test.v` | perf gates: 20k-iteration `sample_into` floor, blocked-vs-naive matvec equivalence + speed |
| `src/infer/backend_test.v` | registry contents, explicit/auto selection, backend agreement |
| `src/formats/formats_test.v` | GGUF/SafeTensors/unknown/PyTorch detection, header parse, dtype table |
| `src/config/config_test.v` | defaults, partial-file fallback, BOM strip, rejection of bad values |

Rules (house policy): every fix ships with a failing-first test where
practical — break the line, watch it go red, restore it. A silent green
proves nothing: confirm the new test names appear in output.

## Conventions (the parts that bite)

* Token loop is allocation-free: hoist scratch buffers out, pass `mut`,
  never `clone()` per token. `sample()` allocates (compat); `sample_into()`
  does not — the loop uses the latter.
* Hot kernels get `[direct_array_access]`; validate shapes at model load,
  never per token.
* `infer` = pure compute (imports `formats` only). No `net`, no `os` wall
  clock in the loop; `os` file access is load-time only and goes through
  `cfile.v` probes on the kernel port.
* `v fmt -w src` before every commit. `v vet` warnings about missing docs
  on `pub` items are real — document them.
* Keep `Gen`/`generate` signatures stable: phase 1b swaps pseudo-logits
  for the transformer forward without touching the server.

## Where to extend

* New weight format → `docs/formats.md` checklist (5 steps).
* New compute target → implement `Backend` in `src/infer/backend.v`,
  register in `list_backends()` + `select_backend()`, add an agreement
  test against `CpuNaiveBackend`.
* New endpoint → `src/api/server.v` route + shape in `src/api/types.v` +
  document in `docs/api.md`. Keep it socket-level: no `veb`.
* New knob → `src/config/config.v` field + default + range check +
  `vllm.example.json` + `docs/configuration.md` row + CLI override in
  `src/main.v` if it deserves a flag.
