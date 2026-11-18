# Security

Threat model first: vllm-os is a single-tenant inference appliance. The
guest runs one statically linked server as pid 1 on a hobby kernel; there
is no SSH, no shell, no second user. Defense therefore concentrates on
the only exposed surface — `:8000` — plus image hygiene. Anything below
assumes the hypervisor/host is trusted (a malicious host reads guest RAM
regardless; that is out of scope until confidential-computing hardware).

## What is enforced today

* **Bearer auth (opt-in).** `server.auth_token` (config file,
  `vllm.example.json` shows the shape). Empty = open mode (default: keeps
  local dev, QEMU gates, and first-boot flows working). When set, every
  endpoint except `GET /health` requires
  `Authorization: Bearer <token>` and answers 401 otherwise. Health stays
  open on purpose: load balancers and the boot gate need an unauthenticated
  liveness signal. Pure function `api.authorized`, unit-tested.
* **Per-IP rate limiting (opt-in).** `server.rate_per_sec` (0 = off) with
  `server.rate_burst` (0 = one second's worth). Token bucket per peer IP,
  mutex-guarded, 429 + `{"error":"rate limited"}` past the budget.
  Protects the (expensive) decode path from accidental floods, not from
  a determined DDoS — that belongs at the edge.
* **Backpressure.** `server.max_conns` (default 64) bounds concurrent
  generations at `accept`; `server.max_body` (default 1 MiB) caps request
  framing before allocation. Both were load-bearing before auth existed.
* **No secrets in the image by default.** The baked `config.json`
  carries no token unless `--auth-token` built it in. Prefer injecting
  the token at deploy time (env `VLLM_API_KEY` → config render) over
  baking it into a shared ISO.

## What is deliberately NOT done in-server

* **TLS termination.** No TLS stack ships in the guest. Terminate at the
  edge (reverse proxy / load balancer with a real cert) and forward
  plaintext to `:8000` over a trusted network. Rationale: a hand-rolled
  TLS in a static userland binary on a hobby kernel is worse than none —
  it invites cipher-suite bugs while looking secure.
* **Sandboxing the server process.** The fork documents pledge/unveil
  hooks in the exec path, but pid 1 cannot drop what the kernel never
  granted: the server legitimately needs net + files + threads, which is
  nearly the whole sandbox. Process isolation arrives with multi-service
  images, not before.

## Guest specifics

* The C init (`kernel/init/init.c`) does one `execve` of `/sbin/vllm`
  and nothing else; the stub listener only runs when the server binary
  is absent. No shell, no `PATH` lookup (absolute argv), no env except
  `VLLM_THREADS=1`.
* `/dev/com1` receives stdout/stderr (dup2 before exec) — logs leak
  prompts/completions to whoever holds the serial console. Treat serial
  output as sensitive in shared environments.
* `tokenizer.json` and weights ride the initramfs in clear. Anyone with
  the ISO has the model. That is accepted (edge weights are public
  artifacts here); do not bake private fine-tunes into shared images.

## Prod checklist (before real users)

1. Bake or inject a strong `auth_token`; verify 401 without it.
2. Set `rate_per_sec`/`rate_burst` to the capacity plan (measure first).
3. TLS + auth at the edge; firewall `:8000` to the proxy only.
4. Pin `max_conns`/`max_body` to the box (defaults fit the 4G guest).
5. Rotate the token per deployment; never reuse a demo token in prod.
