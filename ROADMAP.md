# vllm-os roadmap — Qwen3-1.7B on trimmed Vinix

Each phase has an exit gate. Do not start the next phase until the gate is green.

## Phase 0 — Host skeleton (done ✅)

`v run src` serves OpenAI-compat stub on Windows/Linux.
Gate: `/v1/models` + `/v1/chat/completions` return 200. Green.

## Phase 1 — Host inference (done ✅)

* ✅ GGUF KV + tensor table parser (`gguf.v`), alignment-aware.
* ✅ Dequant kernels Q4_0/Q8_0/F16 (`quant.v`), RMSNorm/SwiGLU/RoPE/matvec (`nn.v`).
* ✅ C AVX2 layer (`vllm_kernels.{h,c}` + `simd_c.v`), runtime-dispatched,
  removable; heap top-k + opt-in fast sampler (`sampler.v`).
* ✅ VTL-aligned zero-copy view (`tensor_view.v`); VTL/VSL parity tests green.
* ✅ Sampler with temp/top-k/top-p + xorshift RNG (`sampler.v`).
* ✅ BPE loader with stub fallback + Qwen chat template (`tokenizer.v`).
* ✅ KV-overridden arch config, wired generate path (`model.v`).
* ✅ `v -prod -cc gcc test src` — 10 files green (api, cfg, formats, infer).
* ✅ Kernels beat NumPy 4/5 rows, tie 5th at DRAM wall (`docs/perf.md`).
* ✅ Phase 1b: transformer forward (attention + KV-cache) for real logits —
  landed: GQA + QK-norm + split-half RoPE decode_step, fused Q4_0/Q4_1/Q6_K
  matvecs (C AVX2, runtime-dispatched), real BPE, wired generate path.
* Gate: perplexity + greedy smoke test vs `llama.cpp` — green: greedy
  tokens identical on the probe prompt, PPL close (see `docs/perf.md`
  for the measured pairs; gate prompts run where weights are present).
* ✅ Real-weights validation (2026-11-17, CPU): `Qwen3-0.6B-Q4_0.gguf`
  (382156480 B, size-asserted) + `tokenizer.json` fetched; golden GGUF
  tests green on the real file; server boots on real weights
  (tensors=310); `hello` → real BPE text at ~2.7 tok/s (stub tokenizer
  emits `<tok_N>` when `tokenizer.json` is absent — expected fallback).

## Phase 2 — QEMU network path (done ✅, 2026-11-17)

* Host server + QEMU guest curls it via SLIRP (`10.0.2.2:8000`).
* `scripts/qemu_test.ps1` / `qemu_test.sh` automated: build-if-missing,
  port poll (no fixed sleep), models + chat asserts, boot→first-token
  timing, exit codes 0/1/2/3, guaranteed cleanup. Host gate green:
  ready=2.6 s, total 2.7 s (< 30 s wall, stub).
* Guest leg validated for real: OpenWrt 25.12.6 x86-64 on SLIRP user
  networking (`-accel whpx -machine q35`), driven via TCP serial through
  the **failsafe shell** (normal boot stalls pre-console on some
  QEMU/host combos — failsafe does not), `10.0.2.15/24` + default route,
  in-guest `wget` of `/v1/models` **and** a full chat round trip:
  GUEST-GATE PASS. ssh path kept as an alternative.
* Found + fixed by this testing: uclient-fetch sends even tiny POSTs as
  `Transfer-Encoding: chunked` with no Content-Length; the server only
  read Content-Length bodies and answered 400. Server now de-chunks
  (capped at `max_body`); chunked and CL POSTs both 200, suite 12/12.
* Gate: cold-boot guest → first token < 30 s wall (stub), log saved —
  green (`qemu/test.log`, `qemu/guest.log`, both gitignored).

Follow-ups (closed 2026-11-18): GGUF-derived identity landed —
boot log and `/health` print the mapped `general.name` /
`general.file_type` labels (`config_from_kv`), falling back to the
1.7B/Q4_K_M defaults only for unmapped codes; `/v1/models` stays the
stable `qwen3-1.7b` id by contract (API + gate scripts assert it).
The 1.7B golden fixture path is covered: `src/infer/gguf_golden_test.v`
asserts both 0.6B and 1.7B tables (size-asserted, skips when `models/`
is absent) and `scripts/fetch_small_models.sh` fetches either file.

## Phase 3 — vinix-mini fork

Fork Vinix, keep only (compile-time selected via `kernel/features.conf`,
every extra default OFF; NVMe/AHCI/e1000/ramfs/DHCP-UDP deleted, not flagged):

```
boot / GDT / IDT / paging / pmm / vmm / kheap
SMP bringup / scheduler / spinlock+mutex
virtio-blk (or NVMe) / virtio-net (or e1000)
TCP/UDP/IP/DHCP / sockets / tarfs / ELF loader
serial log (no framebuffer shell)
```

Drop: framebuffer shell/DE, USB HID, audio, most syscalls, userland shell.
Gate: kernel boots on `qemu-system-x86_64 -display none -serial stdio`,
prints `[vllm-os] boot ok`, answers TCP SYN on :8000.

Status 2026-11-17: fork builds and boots to the init handoff under QEMU
(serial log: ACPI/SMP/scheduler/RNG/lwIP+e1000+DHCP, tmpfs/devtmpfs/
procfs/sysfs mount, initramfs unpack, console/tty/serial; then the
expected `V panic: Could not start init process` — no userland yet).
Toolchain fixes against pinned V `7490a8f` (all in `kernel/mini/`, each
marked `NOTE(vllm-os)`): `uint` is not a builtin (per-module phantom
type) → `u32` internally, `u64` + `C.size_t` at C `size_t` boundaries;
`@type` on C structs (e.g. lwIP `ip_addr`) mangles to `_v_type` →
`c/vinix_ip_type.{h,c}` accessors; `&` on function-pointer values is
dropped (copies callee code) → `c/net_driver_abi.h` store helpers;
`u32(ptr)` address truncation → `u64` cursors; `is <interface>` with
zero implementors compiles to a non-nil check → forced-false fallbacks
(MountSpecifier, RenamePublication, FallibleFileSystemStatResource,
MappingAttributesResource, StatSnapshotResource, MetadataUpdateResource,
PrivateMountOptions); `name.str` map-key closure + fixed-array literal
spill worked around; heap_benchmark sampler dropped (its generator
needs newer-V spread syntax). Trimmed away: aarch64/powerpc/apple/drm/gpu/linuxkpi, ath10k/hda/xhci/
mouse/fbdev/nvme/rtl8169/pointerdev, nfs/smb2/ninep (+top files),
block/verity, agx_fake ×2, heapbench (+its sampler); `main_amd64.v`
unwired to e1000+AHCI+serial, framebuffer stubbed. Kept: ext2 (sysdisk
probe path). Vinix upstream master is
red since ~Nov 11 (same breakage class; PRs #267/#269/#254 unmerged).

Status 2026-11-18: mini init written (`kernel/init/init.c`, raw-syscall
static `/sbin/init`: prints `[vllm-os] boot ok`, listens :8000, stub JSON
per connection; only write/socket/setsockopt/bind/listen/accept/read/
close/openat/exit_group — all in the fork's Linux-compat table). Natively
verified in WSL (static SYSV ELF, no PT_INTERP, HTTP 200 + matching
Content-Length 33). amd64 `/dev/console` is the framebuffer (serial
mirror is a no-op), so init also emits the boot lines to `/dev/com1`
(openat, best effort). Build notes: `-fno-builtin` required (gcc turns
the byte-count loop into an unresolvable strlen under `-nostdlib`).

Gate GREEN 2026-11-18: Limine BIOS+UEFI ISO (xorriso 1.5.6 via dpkg -x,
limine-binary 12.9.3, initramfs.tar carries `./sbin/init` mode 755)
boots under `qemu-system-x86_64 -m 2G -display none -serial file` with
e1000 on SLIRP user-net: serial shows `[vllm-os] boot ok` + `listening
on :8000`, no init panic; host connects via hostfwd, gets HTTP 200
`{"model":"mini-stub","boot":"ok"}` (Content-Length 33 exact), and init
loops back to accept. Flake found + fixed in the harness: the host-side
TCP connect can complete at QEMU's hostfwd layer before the guest gets
its DHCP lease — the gate must wait for `DHCP lease` in the serial log
before connecting. (Repro/tool paths live in WSL: `~/isowork/staging`,
`~/.local/xorroot`, `~/isowork/limine-tool`; not yet committed scripts.)

Reproducible since 2026-11-18: `sh kernel/init/build-image.sh`
(committed) performs the whole assembly — static init build, initramfs
tar+gzip, pinned limine-binary 12.9.3 download (tag `v12.9.3`, asset
`limine-binary.tar.gz`; needs `https_proxy=http://192.168.2.5:10808`
on this LAN), xorriso bootstrap via `apt-get download` + `dpkg -x`
(no sudo), mkisofs + `limine bios-install` — and
`scripts/mini_boot_gate.ps1` runs the QEMU gate (exit 0/1/2).
Script-built ISO booted green end to end (`MINI-BOOT-GATE PASS`).

## Phase 4 — Bare-metal inference

Step 1 done 2026-11-18: a normal V userland binary runs on the fork.
The C init now `execve("/sbin/vllm")` after printing boot ok (falls back
to the :8000 stub only when it is absent); `kernel/init/probe.v`
(GC + strings + `os.write_file` to `/dev/com1`) packed as `/sbin/vllm`
via `build-image.sh --initramfs-extra` prints
`[vllm-os] userland V probe ok` on the guest serial — i.e. fork/execve,
V runtime startup (mmap/brk), GC, and devtmpfs open/write/close all
work for userland V. Build: utility V 0.5.2 (`~/vlinux/v/v`) with
`-cc gcc -cflags '-static'` (V has no `-static` flag).

Step 2 done 2026-11-18: real inference in-guest, bit-identical to host.
`tools/infer_smoke.v` (greedy prefill + 8 decodes, `temp: 0.0`,
`VLLM_THREADS=1`) runs on the host as reference and as `/sbin/vllm` in
the guest (C init passes model path + prompt + n as argv; weights +
tokenizer ride in the initramfs via `--initramfs-extra`, guest boots
with `-m 4G`). Guest serial shows the same tokens as the host run:
`tok_ids=[0, 1096, 374, 264, 4285, 3110, 315, 1246]`,
`! This is a simple example of how`, `INFER-SMOKE-OK`. Output goes
through a stdout + `/dev/com1` mirror (`emit`), since guest stdout is
the headless framebuffer. Build note: `src` needs `-path` on the V
command line (`-path "$R/src|@vlib|@vmodules"`); keep image work dirs
on ext4 (`--work /tmp/...`), 9p is too slow for the 400 MB tar+gzip.
Next: serve loop — `/v1/chat/completions` over the :8000 socket in
`/sbin/vllm` instead of the C stub, still userland.

Step 3 done 2026-11-18, Phase 4 gate GREEN: the full host server
(`src`, built `-prod -cflags -static`) runs as `/sbin/vllm` in the
guest. C init now `dup2`s `/dev/com1` onto fds 1+2 (server logs land
on serial) and execves with
`[/sbin/vllm --model /models/model.gguf]` + `VLLM_THREADS=1`.
QEMU `-m 4G` from the 387 MB ISO: serial shows
`boot ok: Qwen3-0.6B format=gguf tensors=310` + `openai-compat api`,
host gets `/v1/models` → `qwen3-1.7b` and a real chat completion
(`Hello! How can I assist you today`) — non-stub, from the ISO.
Harness notes: python on WSL must bypass the LAN proxy for 127.0.0.1
(`no_proxy`); keep weights in `~/vllm-assets` (`/tmp` is wiped between
logins); `net.listen_tcp(.ip6, …)` accepts IPv4 on the fork's lwIP
dual-stack socket with no changes.

* Build `src/infer` with `-gc none -freestanding`, arena KV-cache.
* mmap GGUF from baked partition (no `os` file APIs in kernel).
* Single accept loop + N worker threads on the trimmed scheduler.
* Gate: QEMU `-m 4G` serves a real (non-stub) completion from the ISO.

## Phase 5 — OS+weights image

Done 2026-11-18, gate GREEN. `make image` (root `GNUmakefile`, WSL)
builds the static server, stages `sbin/vllm` + `models/model.gguf` +
`models/tokenizer.json` + `models/config.json`, and emits Limine
BIOS+UEFI `vllm-os.iso` via `kernel/init/build-image.sh`
(`scripts/build_image.sh` is a thin compat wrapper). `config.json` is
always baked (explicit `--config` file, else a minimal default pointing
at `/models/model.gguf` — `cfg.load` fills the rest); init passes
`--model` + `--config`. Fresh QEMU `-m 4G` boot from `vllm-os.iso`
with no host files: real chat completion served, all serial asserts
green. Outside world sees only `:8000` (OpenAI API, no SSH/shell —
there is no shell to expose).

## Phase 6 — ISO builder (CLI first, GUI stretch)

A guided tool that builds a bootable ISO for custom models: pick target
arch, model weights, and serving config, get a tested `vllm-os.iso`.

* CLI: `scripts/build_iso.py` (interactive + flags). Prompts (defaults
  in brackets): arch (`x86_64`; `arm64` listed but refused until the
  fork's amd64-only drivers are ported — no silent wrong-arch images),
  GGUF path (size-asserted, refuses > guest RAM budget), tokenizer
  sidecar (required, must sit next to the weights), port / ctx cap /
  backend / sampler preset (writes the baked `config.json`; partial
  files OK, `cfg.load` fills the rest). Then drives `make image`
  (MODEL / TOKENIZER / CONFIG / KERNEL passthrough) and optionally
  boots the result under QEMU for a smoke chat.
* GUI: `gui/` — a vlang/ui2 desktop app on the same backend
  (arch dropdown with arm64 refusal, all knobs, Generate + Copy
  buttons, boot command readout). Pure composer (`gui/compose`,
  unit-tested) plus a thin ui2 shell (`gui/main.v`, `--smoke`
  asserts all defaults render). No backend, no uploads, no Electron:
  it composes the exact `build_iso.py` invocation for copy-paste.
  Toolchain exception (documented in docs/development.md): ui2 needs
  json2, so `gui/` builds with a weekly V and stays out of the
  release-pinned CI lanes; `gui/compose` still tests green on the pin.
* Gate: tool-built ISO for a non-default model boots fresh in QEMU and
  serves `/v1/chat/completions` (reuse `scripts/mini_boot_gate.ps1`
  asserts); `--dry-run` prints the exact `make` invocation without
  building.

Status 2026-11-18: CLI landed (`scripts/build_iso.py`, WSL).
Refuses arm64 / missing weights / missing sidecar (unless
`--allow-stub-tokenizer`) / bad backend-preset-port-temp / over-budget
weights, generates `config.json` from knobs (verified through the real
server `--print-config`: `cfg.load` + `validate` green), drives
`make image`, prints the QEMU boot line. Fixture-tested (10 refusal
cases + happy path + stub hatch). GUI is a vlang/ui2 desktop app
(`gui/`, `--smoke` green) replacing the interim HTML page.

Gate closed 2026-11-18: full `build_iso.py` run with non-default
knobs (`--port 8080 --ctx 2048 --preset precise --temp 0.5`) built a
387 MB ISO that fresh-boots and serves real chat on `:8080`
(`openai-compat api on :8080` on serial, `GUEST-SERVE-OK` via hostfwd)
— the generated config demonstrably takes effect in-guest.

## Production hardening (in progress)

Done 2026-11-18: bearer-token auth (opt-in, `/health` stays open),
per-IP token-bucket rate limiting (opt-in), 401/429 framing, unit tests
in the CI file list, `docs/security.md` (threat model, TLS-at-edge,
serial-log hygiene, prod checklist), `--auth-token`/`--rate-*` knobs in
`build_iso.py`. Verified on metal: guest ISO with a baked key returns
401 without it and serves real chat with it (`GUEST-AUTH-OK`).
Architecture note: the gate runs in the single accept-loop thread before
`spawn` — pinned V 0.5.2 miscompiles `spawn` with a shared `mut` ctx
(handler threads died reading garbage; bisected), so ctx stays immutable
and the bucket map needs no locks.

Done 2026-11-18 (serve path): the model blob is cached at boot
(`Model.blob`) instead of re-read per request, and KV arenas are sized
to the request (prompt + max_tokens via `new_forward_ctx_cap`) instead
of the full 4k cap (~900 MB zero-fill per request on the 0.6B shape).
Paired same-box probe (temp 0, seed 7, `Hello`, `chat8`): wall 7.3 s ->
2.9 s (~2.5x), and 1.5 s after the win-12 kernel rewrite
(~4.7x cumulative); greedy tokens identical across all paths. Serve
throughput is prefill-bound (ChatML prompt ~1.2 s) with steady-state
decode ~48-50 ms/token (see `docs/perf.md`).

Closed: TLS terminates at the edge by design (no TLS stack ships in the
guest — `docs/security.md` rationale); SMP is validated by the 1.7B
workflow (`-smp 2` QEMU boot to a real completion, weekly + manual);
1.7B boots via `.github/workflows/model-17b.yml` (golden table + full
ISO + real chat asserts, artifacts kept as serial logs); perf-CI numbers
land since 2026-11-18 (`docs/perf.md` CI section auto-updated, runners
vary so rows track regressions only).

## Phase 7 — llama-modchip takeaways (planned, 2026-11-19)

Source: `synchopate/llama-modchip v3.0` (`llama.cpp @ de32b268c` +
`patches/llama-modchip-v3.0.patch`, CLI `modchip` / `install.sh` /
`scripts/`, `ARCHITECTURE.md`). Rule: ideas only, nothing vendored —
no C++/OpenMP, no Metal/CUDA/Vulkan shaders, no 13–58 GiB weights in
the image. Every item keeps `-gc none`, freestanding-safe kernels, and
`tools/`-only VTL/VSL. License check before copying any C from
`patches/` (vllm-os is Apache-2.0; record provenance per file).

* A — Non-convoy threading. Adopt `threads = phys_cores - 1`,
  one core reserved for the accept loop (`src/api/server.v`), no
  oversubscription, `--poll`-style busy-poll off; then a persistent
  pool to replace spawn-per-call (`src/infer/parallel.v`,
  `cpu-parallel` backend in `src/infer/backend.v`). modchip reports
  +25.9% pp / +31.3% tg from this fix alone. Gate: `tools/forward_bench.v`
  threaded vs serial improves on the same box, `parallel_test.v` +
  `bench_test.v` green, guest stays pinned serial under
  `VLLM_THREADS=1`.
* B — Draftless speculation. Port the idea of `Stockfish-PVS v2` /
  `RouteCrack ngram-simple 3x3` (modchip: 1.67–1.87x cold, 11–14 tok/s)
  as an opt-in ngram drafter in `src/infer/sampler.v` (pure V, no
  extra weights, arena-allocated); built-in MTP stays out (needs
  `blk.48`-style weights we do not ship). Gate: accept-rate metric in
  `forward_bench`, greedy `temp: 0.0` output bit-identical with
  speculation on/off (`in-hw vs host` parity like `infer_smoke.v`).
* C — Low-bit quants + quality gates. Add `Q2_0` signed-scale dequant
  + fused AVX2 matvec next (`src/infer/quant.v` + `vllm_kernels.{h,c}` +
  `simd_c.v`, runtime-dispatched, `VLLM_NO_SIMD=1` fallback); `Fractal-64`
  (type 44, 3.75 bpw) / `VQ3-64` (type 45, 3.0 bpw) only via the
  `ContainerHandler` registry (`src/formats/`, `docs/formats.md`)
  if a model needs them — never as a second native path. Adopt the
  modchip gate trio: blind `wiki-alt` PPL + KLD + Same-top-p plus
  cgroup `VmHWM` enforcement for the guest RAM budget. Gate: real-weight
  maxdiff inside the 1e-3 band, `vtl_parity_test` style parity,
  `docs/perf.md` rows updated (same box only).
* D — Memory/VM. `madvise(MADV_HUGEPAGE)` inward-aligned to 2 MiB for
  the boot-cached blob (`Model.blob`), `_MM_HINT_NTA`-style streaming
  for weight sweeps vs L3-resident KV, keep 3D/MoE tensors zero-copy
  mmap-backed; host-side `/dev/shm` Slot-0 KV snapshot/restore as a
  `tools/` experiment (modchip: ~38 ms vs ~6 s prefill). Gate: same
  greedy tokens, measured wall-time win on `chat8`, no `mmap`/THP
  dependency in the freestanding path.
* E — Long context. Opt-in sliding-window attention + sinks
  (`LLAMA_ATTN_WINDOW`-style knob in `src/cfg/config.v` +
  `src/infer/attention.v`, default exact full attention); hybrid
  SSM/Mamba-style models stay a research note (modchip/Nemotron:
  6 GQA layers = 1.7 KB/token vs 27 KB/token dense — the reason for
  our 4k cap in `docs/architecture.md`). Gate: attention-scaling probe
  (`n=512/4096` head-step) recorded in `docs/perf.md`, 4k default
  unchanged until a ≥16 GB guest exists.

Non-goals (stay deferred): whole patched `llama.cpp`, `GLIBC<=2.30`
portable binary, GPU backends, `*-ablit` weights, `sse_gateway.py` in
the guest (host-side proxy only), 30B/284B weights.

## Phase 8 — similar-engine takeaways (planned, 2026-11-19)

Sources: `ikawrakow/ik_llama.cpp` (IQK + repack + MLA/GDN, late-2026
diverged), `ggml-org/llama.cpp docs/speculative.md` (draft/EAGLE/DFlash/
ngram taxonomy), `Tiiny-AI/PowerInfer(-2)` (hot/cold sparse activation),
`kvcache-ai/KTransformers` (CPU+GPU heterogeneous MoE, AMX),
`microsoft/BitNet.cpp` (ternary 1.58-bit kernels). Same rule as Phase 7:
ideas only, nothing vendored; `-gc none` + freestanding-safe + `tools/`-only
VTL/VSL stay. License/provenance check before copying any C.

* F — IQK + row-interleaved repack (from `ik_llama`). Prefer standard
  `IQ4_NL` (+ `Q2_0_R8` repack variant) over custom types 44/45: add
  `IQ4_NL` dequant + fused AVX2 matvec in `src/infer/quant.v` +
  `vllm_kernels.{h,c}` + `simd_c.v` (runtime-dispatched,
  `VLLM_NO_SIMD=1` fallback), repack as a load-time layout behind
  `Backend` (`src/infer/backend.v`), never a second math path. Note
  the `ik` `-rtr` warning: repack only tensors the CPU really runs.
  Gate: real-weight maxdiff in 1e-3 band, `quant`/`simd`/`bench` tests
  green, `docs/perf.md` same-box rows (pp/tg) move, not just microbench.
* G — Upstream speculative taxonomy (from `llama.cpp`). Ship only
  `ngram-simple/cache` (no draft weights, pure V in `src/infer/sampler.v`,
  arena-allocated, opt-in flag + `VLLM_*` env in `src/cfg/config.v`);
  `EAGLE3/DFlash/DSpark` stay out (need per-target draft GGUFs that do
  not fit the 4 GB guest). Gate: accept-rate + wall-time in
  `tools/forward_bench.v`, greedy `temp: 0.0` bit-identical on/off,
  host-vs-guest parity like `infer_smoke.v`.
* H — Sparse-activation note (from `PowerInfer`). No port: hot/cold
  predictors need ReLU-sparse models, `Qwen3 SwiGLU` is dense. Recorded
  only as model-selection bias: prefer MoE over dense-sparse for future
  models (same conclusion as the Nemotron note in Phase 7-E). Gate:
  docs-only (`docs/architecture.md` one paragraph), no code.
* I — Heterogeneous boundary (from `KTransformers`). Stays deferred:
  CPU+GPU offload with AMX async scheduling beats pure CPU for
  100B+ MoE, but `vinix-mini` has no GPU drivers by design
  (`docs/architecture.md` constraints). Record the boundary so a future
  `cpu-gpu` `Backend` has a seam (`list_backends()` + `select_backend()`).
  Gate: docs-only, no guest code.
* J — Ternary research note (from `BitNet.cpp`). No port in v1: 1.58-bit
  needs trained ternary weights, no `Qwen3` GGUF exists. Keep as the
  sub-2bpw reference point for Phase 7-C quality gates (PPL/KLD/Same-top-p).
  Gate: docs-only.

Non-goals (stay deferred): vendoring `ik_llama`/upstream diffs, CUDA/AMX
kernels in-guest, draft-model GGUFs in the ISO, ReLU-sparse-only models,
ternary weights, distributed/RPC inference (`exo`/`llama.cpp RPC` class).

## Phase 9 — second-wave takeaways (planned, 2026-11-19)

Sources: `apoorvumang/prompt-lookup-decoding` (PLD, now in HF
transformers as `prompt_lookup_num_tokens` + in vLLM as
`speculative_model="[ngram]"`), `EricLBuehler/mistral.rs` (Rust engine:
ISQ, paged attention, prefix caching, chunked prefill, auto-detect),
`intel/ipex-llm` (ARCHIVED by Intel — known security issues, no fixes;
ideas only, never a dependency), `mlc-ai/mlc-llm` (TVM compiler engine),
`karpathy/llm.c` (pure C/CUDA reference, ~1000-line CPU fp32 in one
file). Same rule as Phase 7/8: ideas only, `-gc none` + freestanding-safe
+ `tools/`-only VTL/VSL stay.

* K — PLD parameters for Phase 7-B/8-G (from `prompt-lookup-decoding`).
  Validated numbers: ~2.4x on input-grounded tasks (summarization,
  context-QA), smaller on first-turn chat; never slower than greedy in
  their runs. Start from their hyperparams: `max_ngram_size = 3`,
  continuation length 10, match over prompt + own output (turn-2 code
  editing is our best case). Implement in `src/infer/sampler.v` (pure V,
  arena-allocated, opt-in). Gate: accept-rate + wall-time in
  `tools/forward_bench.v` on a code-edit prompt, greedy `temp: 0.0`
  bit-identical on/off.
* L — Serve-path trio (from `mistral.rs`). Three ideas that attack our
  measured prefill-bound serve wall (`chat8` ~1.5 s, ChatML prompt ~1.2 s):
  prefix caching (reuse prompt KV across turns), chunked prefill (bound
  per-request prefill work), ISQ — in-situ quantization on load
  (host-side only, e.g. in `scripts/fetch_model.sh` flow, never in the
  guest). Paged attention stays a research note (our flat arena in
  `src/infer/attention.v` is simpler and already request-sized). Also
  adoptable: hardware-aware `tune` (print recommended
  backend/threads/quant from detected CPU, like `mistralrs tune`) as a
  `--tune` host flag. Gate per item: same greedy tokens, wall-time win
  on `chat8`, `vllm.example.json` + `docs/configuration.md` row for any
  new knob.
* M — IPEX-LLM boundary (ARCHIVED — never depend). Takeaways only:
  low-bit CPU path (INT4/FP8/FP6/FP4), self-speculative decoding (~30%
  reported on CPU), FlashMoE for 671B-class on small GPUs; plus their
  mmap side-channel warning (`--no-mmap` for multi-tenant) — record in
  `docs/security.md` that our mmap'd `Model.blob` is safe only because
  the appliance is single-tenant by design. Gate: docs-only.
* N — Compiler-vs-handkernels (from `mlc-ai/mlc-llm`). TVM compiles one
  source to CPU/CUDA/Metal/Vulkan/WebGPU with an OpenAI-compat server on
  every target — the portability ideal — but it needs a compiler runtime
  no freestanding guest can host. Rejected for the token loop with
  reason; keep hand-written `vllm_kernels.c` (style precedent: `llm.c`
  below). Host-side autotuning stays an open question for `tools/`.
  Gate: docs-only.
* O — Single-file C precedent (from `karpathy/llm.c`). `train_gpt2.c`
  proves a whole fp32 transformer fits in ~1000 clean dependency-free C
  lines with `OMP_NUM_THREADS` threading — the same shape as our
  `vllm_kernels.{h,c}` (no libm/libc in the loop, OMP optional). Use as
  the style bar for every new kernel in Phase 7/8: one file, explicit
  loops, software f16 (we already do), no hidden allocator. Gate:
  review checklist, no new dependency in any kernel PR.

Non-goals (stay deferred): draft-model weights (EAGLE/DFlash class),
GPU/NPU runtimes, TVM/compiler toolchain in the image, training code,
multi-tenant mmap exposure.

## Deferred (deliberately out of v1)

GPU / NPU drivers, multi-tenant auth + rate limits, full 32k ctx,
quant servers beyond Q4_K_M/Q8_0, ARM64 bare metal, OTA updates.
