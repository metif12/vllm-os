# Configuration reference

Source: `--config vllm.json` (see `vllm.example.json`). Precedence:
**CLI flag (when != default) > config file > built-in defaults.**
Partial files are valid — missing fields normalize to defaults, then the
whole config validates. A UTF-8 BOM (Notepad) is stripped.

## `server`

| key | default | valid | notes |
|---|---|---|---|
| `host` | `0.0.0.0` | any bind addr | kernel port binds the static IP instead |
| `port` | `8000` | 1–65535 | baked into the ISO service record |

CLI: `--port` (default 8000; overrides file only when explicitly set).

## `model`

| key | default | valid | notes |
|---|---|---|---|
| `weights` | `models/qwen3-1.7b-q4_k_m.gguf` | existing file | missing file → stub mode (HTTP up, no weights) |
| `format` | `auto` | `auto\|gguf\|safetensors` | `auto` sniffs magic bytes; `.bin` gets a convert-first error |
| `ctx_cap` | `4096` | 128–131072 | KV arena sizing; 4k fits the 4 GB guest (see `kernel/mini.md`) |
| `backend` | `cpu` | `cpu\|cpu-blocked` | `cpu-blocked` wins on wide projections; `auto` picks by shape in phase 1b |

CLI: `--model`, `--ctx` (same override rule as `--port`).

## `api`

| key | default | valid | notes |
|---|---|---|---|
| `max_tokens` | `256` | 1–32768 | per-request cap; generation loop additionally clamps to 64 until the real forward lands |

## Example

```json
{
  "server": { "host": "0.0.0.0", "port": 8000 },
  "model": {
    "weights": "models/qwen3-1.7b-q4_k_m.gguf",
    "format": "auto",
    "ctx_cap": 4096,
    "backend": "cpu"
  },
  "api": { "max_tokens": 256 }
}
```

```sh
./server --config vllm.json
./server --config vllm.json --port 9000   # file for everything, flag for the port
```
