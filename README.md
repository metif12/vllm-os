# vllm-os — bare-metal LLM appliance in V

Boot straight into a model. No Python, no console/GUI in production.
`OS + weights` image exposing an OpenAI-compatible API on `:8000`.

Target v1 model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) → GGUF `Q4_K_M` (~1.1 GB).

## What this is

A trimmed OS (Vinix fork: net + block + minimal scheduling only) that
boots directly into an inference loop. Outside the box it looks like
OpenAI: `GET /v1/models`, `POST /v1/chat/completions` (incl. SSE stream),
`POST /v1/completions`. Weights ship inside the boot image.

There is intentionally no Python runtime, no shell, and no desktop
in the image. Host-side Python exists only for one-off weight conversion
(`safetensors → GGUF → quant`) and is never shipped.

## Layout

```
src/main.v            # boot entry: load weights -> serve API
src/infer/gguf.v      # GGUF header parser (no full-file read)
src/infer/cfile.v     # tiny C file probe (header only, 1 GB-safe)
src/infer/model.v     # Qwen3 arch config + weights handle
src/infer/sampler.v   # greedy / top-k / top-p sampler (pure V)
src/infer/tokenizer.v # stub + Qwen ChatML template (BPE port = phase 2)
src/api/server.v      # raw-socket HTTP, no veb (bare-metal portable)
src/api/types.v       # OpenAI JSON shapes
kernel/               # vinix-mini fork plan
scripts/              # host tooling only (fetch/convert/qmp test)
qemu/                 # QEMU harness docs
ROADMAP.md            # phased plan with exit criteria
```

## Quick start (host, no QEMU yet)

```sh
v run src --model models/qwen3-1.7b-q4_k_m.gguf --port 8000
curl localhost:8000/v1/models
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi"}]}'
```

No weights file yet? The server still boots in stub mode so you can
test the HTTP surface before downloading gigabytes.

## QEMU test

See `qemu/README.md`. Requires `qemu-system-x86_64` on PATH.
Windows: `winget install -e --id SoftwareFreedomConservancy.QEMU`.

```sh
v -o server.exe src
./scripts/qemu_test.ps1
```

Phase-1 QEMU test proves the guest→host network path: server runs on
the host, a tiny QEMU guest curls `/v1/models` and `/v1/chat/completions`
over the SLIRP link (`10.0.2.2`). The bootable `vinix-mini` ISO is phase 3.

## Model weights (not committed)

```sh
./scripts/fetch_model.sh
```

Downloads `Qwen3-1.7B` safetensors, converts with `llama.cpp`
`convert_hf_to_gguf.py`, quantizes to `Q4_K_M`, drops the result in
`models/`. The OS image build bakes `models/*.gguf` + `tokenizer.json`
into the boot partition.

## API

* `GET /v1/models` → `{"object":"list","data":[{"id":"qwen3-1.7b",...}]}`
* `POST /v1/chat/completions` → chat completion, `stream:true` gives SSE
* `POST /v1/completions` → legacy completion

Qwen3 chat template (`<|im_start|>…`) is applied server-side.

## Design constraints

* Inference is pure V + `C` SIMD intrinsics, `-gc none`, arena KV-cache.
  No GC pauses inside the token loop on bare metal.
* HTTP is hand-rolled on `net` sockets — `veb` needs OS syscalls Vinix-mini
  does not provide yet.
* Context is capped at 4k tokens in v1 (Qwen3 native is 32k; full 32k KV
  does not fit the v1 RAM budget).
* RAM budget: ~1.1 GB weights + KV + server → 4 GB guest minimum.

## License

Apache-2.0 (matches Qwen3-1.7B licensing for the weights flow).
Kernel fork keeps upstream Vinix license notices — see `kernel/`.
