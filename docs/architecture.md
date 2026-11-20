# Architecture

One sentence: a trimmed OS boots straight into a token loop and looks like
OpenAI from the outside.

## Boot flow

```
firmware → vinix-mini → serial init → pmm/vmm → virtio-blk → mmap weights
  → virtio-net (static IP) → scheduler + workers → :8000 → [vllm-os] boot ok
```

No login, no shell, no userland. The kernel command line
(`kernel/cmdline.txt`) carries `model=`, `port=`, `ctx=`, `ip/gw/netmask`,
`workers=`, `log=serial`. The runner reads the same keys from `vllm.json`
on the host path.

## Request flow (host build today, kernel build tomorrow)

```
TCP :8000 → read_request (headers + exact Content-Length)
  → route (models | chat | completions)
  → apply_chat_template (<|im_start|>…; Qwen3 ChatML)
  → tok.encode (BPE or stub) → sample loop → escape_json → SSE or JSON
```

HTTP is hand-rolled on `net` sockets (`src/api/server.v`), not `veb`:
`veb` needs syscalls vinix-mini does not provide. One consequence is
deliberate: `Connection: close` everywhere, no keep-alive state machine
in the kernel port.

## Module map

| module | owns | never touches |
|---|---|---|
| `src/main.v` | CLI + `--config`, env layer, precedence, introspection flags, boot logging | tensor bytes |
| `src/cfg` | `vllm.json` + `VLLM_*` load/normalize/validate, preset resolution | network, weights |
| `src/formats` | container sniffing + `ContainerHandler` registry, SafeTensors index | inference math |
| `src/infer` | GGUF index, kernels, sampler + presets, BPE, `Tokenizer` iface, `Model` | sockets |
| `src/infer` → `backend.v` | `Backend` interface + registry (`cpu`/`cpu-blocked`/`cpu-simd`/`cpu-parallel`/`auto`) + introspection | (same module, seam for SIMD/GPU) |
| `src/infer` → `parallel.v` | threaded row-block matvec over the dispatched kernel (`VLLM_THREADS`, freestanding-safe serial fallback) | OS thread pool, allocation |
| `src/infer` → `attention.v` | phase-1b primitives: `KVCache` arena + `attention_step` | sockets, weights I/O |
| `src/infer` → `simd_c.v` + `vllm_kernels.{h,c}` | C AVX2+FMA kernels + V wrappers, runtime-dispatched, removable (non-prod links none) | OS calls, libm, allocation |
| `src/infer` → `tensor_view.v` | VTL-aligned zero-copy Tensor view (shape/strides/reshape/matvec), no dependency | data ownership |
| `src/api` → `routes.v` | `Route` table + `match_route` (new endpoint = one row here) | sockets |
| `src/api` → `handlers.v` | per-endpoint logic + sampler merging, SSE framing | sockets |
| `src/api` → `limits.v` | bearer-auth check + per-IP token-bucket gate (accept-loop thread only, no locks) | sockets |
| `src/api` → `server.v` | socket loop, request framing, response bytes | weight blobs, JSON shapes |
| `src/api` → `types.v` | OpenAI JSON shapes | weight blobs |
| `tools/` | host-only dev tools (`bench_kernels.v` kernel rows, `forward_bench.v` full-decode probe, `infer_smoke.v` greedy host-vs-guest parity); link VTL/VSL freely — never ship in the image | production paths |

Dependency direction: `main → api → infer → formats`, `main → cfg`,
`api → cfg`. `infer` never imports `api`; `formats` never imports
`infer`. Extension points (each = one file + registry row, never a
dispatch rewrite):

* New weight format → implement `formats.ContainerHandler`, append to
  `default_handlers()` (`docs/formats.md` checklist). GGUF stays native.
* New compute target → implement `Backend` in `src/infer/backend.v`,
  register in `list_backends()` + `select_backend()`.
* New sampler preset → add to `preset_params()` in `src/infer/sampler.v`;
  config validation picks it up via `infer.sampler_presets()`.
* New tokenizer → implement `infer.Tokenizer` (`encode` + `name`); the
  generate path and server need no changes.
* New endpoint → `types.v` shape + `handlers.v` fn + one `routes()` row.
* New knob → `src/cfg/config.v` field + default + range check +
  `vllm.example.json` + `docs/configuration.md` row + CLI override in
  `src/main.v` if it deserves a flag (+ `VLLM_*` env in `apply_env`).

The C layer is optional by construction: non-prod builds exclude it,
`VLLM_NO_SIMD=1` and the `backend` knob disable it at runtime, and
deleting `simd_c.v` + `vllm_kernels.{h,c}` + the `CpuSimdBackend`
block leaves a pure-V tree.

VTL/VSL live in `tools/` and `*_test.v` parity only — never in the
token loop (GC-managed tensors violate the `-gc none` contract).
`tensor_view.v` mirrors VTL naming so a host reference path can swap
to real VTL without signature changes; `vtl_parity_test.v` proves the
mirror element-for-element.

## Load-bearing constraints

* Token loop is allocation-free: scratch buffers (`logits`, `idx`,
  KV-cache arena) are hoisted out and reused. `-prod` strips asserts;
  the kernel port builds `-gc none -freestanding`.
* Hot kernels carry `[direct_array_access]`; shapes are validated once at
  model load, never per token.
* Context cap 4k is a RAM decision, not a model decision: full 32k KV for
  Qwen3-1.7B exceeds the 4 GB guest budget (~2 GB over). Raise only on
  ≥16 GB guests.
* Full design numbers live in `kernel/mini.md` (boot order, memory budget,
  gates).

## Engine boundaries (docs-only, Phase 10)

These engines were evaluated and deliberately kept out of the guest; they
are recorded here so the decision is not re-litigated:

* **PowerInfer / PowerInfer-2.** Hot/cold weight partitioning needs
  ReLU-sparse models to get its activated-neuron savings. Qwen3 uses dense
  SwiGLU, so the sparsity premise does not hold; the same trick would only
  pay on a different model family. Kept as the sub-2bpw reference point
  alongside BitNet, not as a runtime path.
* **KTransformers.** CPU+GPU offload requires GPU drivers vinix-mini does
  not ship. The seam stays at `list_backends()` / `select_backend()`:
  another backend can be added there without touching model or server code,
  but there is no guest-side GPU to target.
* **mlc-ai/mlc-llm.** TVM needs a compiler runtime (codegen, module loading)
  that a freestanding `-gc none` guest cannot host. Rejected with that
  reason, not for lack of kernels.
* **BitNet.cpp.** 1.58-bit ternary needs weights trained with ternary
  quantization; converting Q4 weights does not produce a valid model. It
  remains the sub-2bpw reference for the Phase 7 quality gates, not a quant
  type we can add to an existing GGUF.
