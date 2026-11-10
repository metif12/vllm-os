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
| `src/main.v` | CLI + `--config`, precedence, boot logging | tensor bytes |
| `src/config` | `vllm.json` load/normalize/validate | network, weights |
| `src/formats` | container sniffing, SafeTensors index | inference math |
| `src/infer` | GGUF index, kernels, sampler, BPE, `Model` | sockets |
| `src/infer` → `backend.v` | `Backend` interface + registry | (same module, seam for SIMD/GPU) |
| `src/api` | HTTP parse, routes, SSE framing, escaping | weight blobs |

Dependency direction: `main → api → infer → formats`, `main → config`.
`infer` never imports `api`; `formats` never imports `infer`. A new weight
format lands in `src/formats/` + one dispatch arm in `infer.load_model`.
A new compute target implements `Backend` and registers in
`list_backends()` + `select_backend()`.

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
