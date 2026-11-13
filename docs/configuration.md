# Configuration reference

Source: `--config vllm.json` (see `vllm.example.json`). Precedence:
**CLI flag (when != default) > `VLLM_*` env > config file > built-in
defaults.** Partial files are valid — missing fields normalize to
defaults, then the whole config validates. A UTF-8 BOM (Notepad) is
stripped.

```sh
./server --print-config          # dump effective config as JSON, exit
VLLM_PORT=9000 ./server          # env override, no file edit
./server --config vllm.json --backend cpu-simd
```

## `server`

| key | default | valid | notes |
|---|---|---|---|
| `host` | `0.0.0.0` | any bind addr | kernel port binds the static IP instead |
| `port` | `8000` | 1–65535 | baked into the ISO service record |
| `max_conns` | `64` | 1–4096 | backpressure: accept waits past this, no unbounded spawn |
| `max_body` | `1048576` | 4096–67108864 | rejects absurd Content-Length before allocation |

CLI: `--port` (default 8000; overrides file only when explicitly set).
Env: `VLLM_HOST`, `VLLM_PORT`, `VLLM_MAX_CONNS`, `VLLM_MAX_BODY`.

## `model`

| key | default | valid | notes |
|---|---|---|---|
| `weights` | `models/qwen3-1.7b-q4_k_m.gguf` | existing file | missing file → stub mode (HTTP up, no weights) |
| `format` | `auto` | `auto\|gguf\|safetensors` | `auto` sniffs magic bytes; `.bin` gets a convert-first error |
| `ctx_cap` | `4096` | 128–131072 | KV arena sizing; 4k fits the 4 GB guest (see `kernel/mini.md`) |
| `backend` | `cpu` | `cpu\|cpu-blocked\|cpu-simd\|cpu-parallel\|auto` | `cpu-simd` = C AVX2 (runtime-degrades without AVX2); `cpu-parallel` = threads + SIMD for big layers (tune with `VLLM_THREADS`, validate with `=1` for serial); `auto` picks by shape; phase-1b forward calls per layer |

CLI: `--model`, `--ctx`, `--backend` (same override rule as `--port`).
Env: `VLLM_MODEL`, `VLLM_FORMAT`, `VLLM_CTX`, `VLLM_BACKEND`.
`VLLM_THREADS` (1–64, default 4) sizes the `cpu-parallel` worker pool;
`VLLM_NO_SIMD=1` forces pure-V kernels at runtime.

Backend resolution is logged at boot
(`backend=cpu resolved=cpu simd=true`); `VLLM_NO_SIMD=1` forces pure-V
kernels at runtime. Introspect without serving:
`./server --list-backends`, `./server --list-formats`.

## `sampler`

| key | default | valid | notes |
|---|---|---|---|
| `preset` | `balanced` | `deterministic\|precise\|balanced\|creative` | base temp/top-k/top-p; explicit fields below win |
| `temp` | `0.7` | 0.0–2.0 | 0 = greedy; per-request `temperature` overrides when set |
| `top_k` | `40` | 0–200000 | 0 = full vocab; partial selection (heap past 256 logits) |
| `top_p` | `0.9` | (0, 1] | nucleus truncation after top-k |
| `min_p` | `0.0` | [0, 1) | 0 = off; drop below `min_p × top-1` prob; top-1 survives |
| `seed` | `0` | any u64 | 0 = per-request seed wins; request 0 + config 0 = time-based |
| `fast` | `false` | bool | approx-exp fast path (sampling only, ~5x at vocab scale) |

Env: `VLLM_PRESET`, `VLLM_TEMP`, `VLLM_TOP_K`, `VLLM_TOP_P`,
`VLLM_MIN_P`, `VLLM_SEED`, `VLLM_FAST=1`.

Preset values: `deterministic` (temp 0), `precise` (0.3/20/0.9),
`balanced` (0.7/40/0.9 = defaults), `creative` (1.1/100/0.95).
Resolution: preset base first, then any sampler field that differs from
its struct default overrides it. The server merges per-request knobs on
top: request `temperature`/`top_p`/`top_k` win when they differ from the
request-struct defaults, else the resolved config stands; `seed` 0 falls
back to config, then to time.

## `api`

| key | default | valid | notes |
|---|---|---|---|
| `max_tokens` | `256` | 1–32768 | per-request cap; generation loop additionally clamps to 64 until the real forward lands |

Env: `VLLM_MAX_TOKENS`. Effective per-request tokens =
`min(request.max_tokens, api.max_tokens)`, minimum 1.

## Example

```json
{
  "server": { "host": "0.0.0.0", "port": 8000, "max_conns": 64, "max_body": 1048576 },
  "model": {
    "weights": "models/qwen3-1.7b-q4_k_m.gguf",
    "format": "auto",
    "ctx_cap": 4096,
    "backend": "cpu"
  },
  "sampler": {
    "preset": "balanced",
    "temp": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.0,
    "seed": 0, "fast": false
  },
  "api": { "max_tokens": 256 }
}
```

```sh
./server --config vllm.json
./server --config vllm.json --port 9000   # file for everything, flag for the port
VLLM_PRESET=creative VLLM_PORT=9000 ./server --config vllm.json
```
