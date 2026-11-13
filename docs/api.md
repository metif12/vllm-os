# API reference (OpenAI-compatible subset)

Base URL: `http://<host>:8000`. `Content-Type: application/json`.
Body cap and connection limit come from config (`server.max_body`,
`server.max_conns`): larger `Content-Length` is rejected before
allocation; excess connections wait at accept (backpressure, no
unbounded thread spawn).

## `GET /health`

```sh
curl localhost:8000/health
```

```json
{"status":"ok","model":"Qwen/Qwen3-1.7B"}
```

Liveness + served model id. No weights touched — used by
orchestration and the QEMU boot gate.

## `GET /v1/models`

```sh
curl localhost:8000/v1/models
```

```json
{"object":"list","data":[{"id":"qwen3-1.7b","object":"model","owned_by":"vllm-os"}]}
```

## `POST /v1/chat/completions`

```sh
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-1.7b",
  "messages": [{"role":"user","content":"hi"}],
  "max_tokens": 256, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
  "seed": 0, "stream": false
}'
```

Fields: `model` (informational, echoed), `messages[]` (`role` + `content`),
`max_tokens` (clamped to 64 in the current pseudo-logit path, full range
with the phase-1b forward; also capped by config `api.max_tokens`),
`temperature` (0 = greedy), `top_k`, `top_p`, `seed` (0 = config seed,
config 0 = time-based; nonzero reproduces generations exactly),
`stream`. Request `temperature`/`top_k`/`top_p` win when they differ from
the request defaults (0.7/40/0.9); otherwise the configured sampler
(preset + explicit fields) stands. The server applies the Qwen3 ChatML
template (`<|im_start|>role\n…<|im_end|>`) before encoding.

Response (`object: chat.completion`): one `choices[0].message.content`,
`finish_reason: stop`, `usage` zeros (token accounting lands with the
real forward).

### Streaming

`"stream": true` returns `text/event-stream`:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…,"choices":[{"index":0,"delta":{"content":"…"},"finish_reason":null}]}
<blank>
data: [DONE]
<blank>
```

One chunk today (the full text), then `[DONE]` — the framing is stable so
clients written against the stream work unchanged when per-token SSE
arrives.

## `POST /v1/completions`

Legacy endpoint: `{"model","prompt","max_tokens","temperature",
"top_p","top_k","seed"}` →
`{"object":"text_completion","choices":[{"text":…}]}`.
Same sampler merge rules as chat.

## Errors

| code | body | cause |
|---|---|---|
| 400 | `{"error":"bad chat request"}` / `bad completion request` | undecodable JSON |
| 404 | `{"error":"not found, …"}` | unknown path (try `/health` or `/v1/models`) |
| — (close) | — | body > `max_body`, or malformed headers |

No auth in v1. The box exposes only `:8000`; there is no SSH or shell.
