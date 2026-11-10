# vllm-os — bare-metal LLM appliance in V

Boot straight into a model. No Python, no console/GUI in production.
`OS + weights` image exposing an OpenAI-compatible API on `:8000`.

Target v1 model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) → GGUF `Q4_K_M` (~1.1 GB).

## Docs

| guide | answers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | boot flow, request flow, module map, constraints |
| [docs/api.md](docs/api.md) | endpoints, SSE, errors, limits |
| [docs/configuration.md](docs/configuration.md) | every knob, precedence, validation |
| [docs/formats.md](docs/formats.md) | GGUF/SafeTensors/PyTorch, conversion, adding a format |
| [docs/building.md](docs/building.md) | profiles, host run, QEMU test, ISO build |
| [docs/development.md](docs/development.md) | test suite, conventions, where to extend |
| [ROADMAP.md](ROADMAP.md) | phased plan with exit gates |
| [kernel/mini.md](kernel/mini.md) | vinix-mini trim spec, memory budget, boot gates |

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
src/main.v            # boot entry: config -> detect format -> serve API
src/config/           # vllm.json loader: server/model/api knobs + validation
src/formats/          # weight containers: detect + SafeTensors index + shared meta
src/infer/gguf.v      # GGUF header + KV + tensor table (no blob load)
src/infer/cfile.v     # tiny C file probe (header only, 1 GB-safe)
src/infer/model.v     # Qwen3 arch config + KV override + generate path
src/infer/sampler.v   # fused softmax-scale, zero-alloc top-k/top-p + xorshift RNG
src/infer/tokenizer.v # BPE loader (tokenizer.json) + Qwen ChatML template
src/infer/quant.v     # Q4_0/Q8_0/F16 dequant + fused Q8 dot (no materialize)
src/infer/nn.v        # rmsnorm/rope-tables/swiglu/unrolled+blocked matvec
src/infer/backend.v   # Backend interface + registry (cpu/cpu-blocked/auto)
src/api/server.v      # raw-socket HTTP + backpressure, no veb (bare-metal portable)
src/api/types.v       # OpenAI JSON shapes
docs/                 # architecture/api/configuration/formats/building/development
kernel/               # vinix-mini fork plan, mini.md spec, mini.ld, cmdline.txt
scripts/              # fetch/build profiles (ps1+sh)/qemu test (host-only tooling)
qemu/                 # QEMU harness docs
vllm.example.json     # config starting point
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

## Configuration

Copy `vllm.example.json` to `vllm.json` and pass `--config`
(CLI flags win over the file; partial files valid; BOM tolerated):

```sh
v -prod -o server src
./server --config vllm.json
./server --config vllm.json --port 9000
```

Full knob reference: [docs/configuration.md](docs/configuration.md).

## Weight formats

GGUF (full index parse), SafeTensors (header-only index), PyTorch
(detected, convert-first error). Details + conversion + the 5-step
new-format checklist: [docs/formats.md](docs/formats.md).

## Build profiles

```powershell
.\scripts\build.ps1 -Profile host-release   # windows daily driver
```

```sh
sh scripts/build.sh host-release   # linux/mac
sh scripts/build.sh bare-metal     # experimental freestanding kernel-port elf
```

Profiles: `host-debug` / `host-release` / `qemu` / `bare-metal`.
Full matrix: [docs/building.md](docs/building.md).

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
Full reference with streaming framing and error table:
[docs/api.md](docs/api.md).

## Status

| area | state |
|---|---|
| Host API (stub + config + formats) | done, tested |
| Inference kernels (dequant/nn/sampler/BPE/backend) | done, bench-gated |
| QEMU network path | harness green (loopback); guest SLIRP via `$GUEST_IMG` |
| Transformer forward (real logits) | next — phase 1b, signatures frozen |
| vinix-mini fork / bare-metal port / ISO | scaffolded (`kernel/`, `scripts/build_image.sh`) |

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
