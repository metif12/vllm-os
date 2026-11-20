# vllm-os roadmap — portable V CPU inference core + thin appliance builder

Goal (redefined 2026-11-19): a portable CPU inference core in V
(`src/infer`, freestanding-safe, `-gc none`) plus a thin appliance
builder that bakes it into a bootable OpenAI-compat image. The v1
reference appliance serves Qwen3-1.7B; the core is model-agnostic.
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

## Phase 7 — fast core (planned, 2026-11-19)

Goal: close the ~3x gap to `llama.cpp` on the same file/box (ours
~48-50 ms/token threaded vs ~16 ms, `docs/perf.md`). Engine work first,
everything else after.

Sources: `synchopate/llama-modchip v3.0` (non-convoy threading),
`ikawrakow/ik_llama.cpp` (IQK + row-interleaved repack), `karpathy/llm.c`
(single-file dependency-free C as the style bar). Rule: ideas only,
nothing vendored; `-gc none` + freestanding-safe + `tools/`-only VTL/VSL;
license/provenance check before copying any C (Apache-2.0).

Status 2026-11-19: the **pool** half landed — `src/infer/worker_pool.v`
gives the decode fork-joins a boot-wide fixed worker set (per-batch
WaitGroups, no cross-request barrier), replacing ~140 spawn-per-call
threads/token. Wired through `Model.pool` → `ForwardCtx.pool` (no global
state); nil under `-freestanding` / `VLLM_THREADS=1`, so the guest stays
serial. Correctness gates green in dev + `-prod` (serial agreement,
bit-exact 2-way/3-way, four-submitter concurrency). The `phys_cores - 1`
default is **deferred**: V 0.5.2 exposes only logical CPUs, and a clean
A/B needs a quiet box (the measurement host was loaded — see
`docs/perf.md` win 13). `parallel_threads()` keeps the measured default of
4 with a `VLLM_THREADS` override until that A/B exists.

Status 2026-11-19 (quants): **IQ4_NL landed** — pure-V dequant/fused dot
(`quant.v`) + AVX2 `pshufb` codebook matvec (`vllm_kernels.{h,c}`), the
`simd_c.v` bridge, threaded + pooled row-block head, and GGUF 18-byte
block sizing. The 16 int8 codebook values and split-half layout are pinned
to the GGML primary source (`ggml-common.h` / `ggml-quants.c`); tests cover
codebook identity, fused-vs-dequant, C-vs-V agreement, threaded agreement,
and tail rows. No IQ4_NL weights ship with the v1 assets, so validation is
synthetic — real-weight maxdiff / PPL awaits an IQ4_NL GGUF.

Status 2026-11-19 (Q2_0): **landed** — GGML type 42, 18-byte/64-elem signed-
scale block, pure-V dequant/fused dot/matvec/embed, AVX2 C matvec (local
dequant scratch + FMA, scalar fallback), threaded + pooled head, dispatch
and GGUF sizing. The **R8 repack** expands 2-bit codes to one signed byte
each at load time so the dot reuses the existing int8/Q8 kernel — no second
math path, host-side only, caller chooses which tensors to repack (the
`ik -rtr` warning). Same synthetic-validation caveat as IQ4_NL: no Q2_0
weight ships in v1.

Status 2026-11-19 (quality gates): **landed** — `tools/quality_gates.v`
runs a fixed in-tree `wiki-alt` probe through the real forward and reports
PPL, KLD(ref||ours) over the reference top-k support, Same-top-p set
agreement, max |Δp|, and VmHWM (Linux; current RSS on Windows), with
`--max-kld` / `--max-topp-miss` / `--max-diff` / `--max-rss-mb` thresholds
and exit codes 0/1/2. A baseline file is plain JSON of top-k distributions:
from a trusted engine (llama.cpp) or from a serial run of this tool. Measured
on Qwen3-0.6B-Q4_0 (loaded host): PPL 39.176 identical serial vs threaded,
KLD 0 / Same-top-p 1.0 / maxdiff 0 threaded-vs-serial, VmHWM ~446 MB serial
and ~485 MB threaded (both far under the 4 GB guest budget). cgroup
`VmHWM` enforcement is CI-side (wrap the tool in a memory-limited cgroup);
the tool reports and enforces the ceiling itself. Phase 7 complete.

* Threads + pool. `threads = phys_cores - 1`, one core reserved for the
  accept loop (`src/api/server.v`), no oversubscription, no busy-poll;
  then a persistent pool replacing spawn-per-call (`src/infer/parallel.v`,
  `cpu-parallel` in `src/infer/backend.v`). Reference: modchip +25.9% pp
  / +31.3% tg from this fix alone; `llm.c` uses `OMP_NUM_THREADS`.
* Quants. `IQ4_NL` first, then `Q2_0` signed-scale (+ `Q2_0_R8` repack
  variant) — standard types over modchip custom 44/45. Dequant + fused
  AVX2 matvec in `src/infer/quant.v` + `vllm_kernels.{h,c}` + `simd_c.v`
  (runtime-dispatched, `VLLM_NO_SIMD=1` fallback); repack as a load-time
  layout behind `Backend`, never a second math path. Heed the `ik`
  `-rtr` warning: repack only tensors the CPU really runs. Style bar:
  `llm.c` — one file, explicit loops, software f16 (we already do), no
  hidden allocator, no new dependency in any kernel PR.
* Quality gates. Blind `wiki-alt` PPL + KLD + Same-top-p plus cgroup
  `VmHWM` enforcement for the guest RAM budget (from modchip).
* Gate: real-weight maxdiff in 1e-3 band, `quant`/`simd`/`parallel`/
  `bench` tests green, `tools/forward_bench.v` threaded vs serial
  improves on the same box, `docs/perf.md` rows move (same box only),
  guest stays pinned serial under `VLLM_THREADS=1`.

## Phase 8 — cheap speculation (planned, 2026-11-19)

Goal: draftless decode speedup with zero extra weights.

Sources: modchip `Stockfish-PVS v2` / `RouteCrack ngram-simple`
(1.67–1.87x cold), `llama.cpp docs/speculative.md` taxonomy,
`apoorvumang/prompt-lookup-decoding` (validated ~2.4x on
input-grounded tasks; now in HF transformers as
`prompt_lookup_num_tokens` and vLLM as `speculative_model="[ngram]"`).

Status 2026-11-19: **correctness foundation landed** (`src/infer/pld.v`).
Opt-in PLD (`VLLM_PLD=1`, `pld_ngram`, `pld_continuation`, greedy only)
finds the longest n-gram suffix match over prompt + own output, proposes up
to 10 drafts, and verifies each with the real forward against the greedy
argmax. Every emitted token is the model's argmax at its position, so output
is bit-identical on/off (gated on real weights). `tools/forward_bench.v`
with `VLLM_PLD=1` reports draft accept rate.

The **wall-time half of the gate is still unmet**. A shared-weight batched
verifier was built (`vllm_q{4_0,4_1,6_k}_matmul_f32`, `decode_batch`, KV
rollback on rejection, `tools/pld_bench.v`) but is **EXPERIMENTAL / opt-in**
via `VLLM_PLD_BATCH=1`: its kernels do a horizontal sum per block per token,
so on the probe they were ~2x slower than the row-vector single-token path,
and their accumulation order can flip a near-tie argmax on accepted drafts.
The default stays bit-exact serial verification. A SIMD-across-k matmul
(k register accumulators, one hsum per row) was then added; it improves the
kernels but end-to-end batching remains slower on every tested prompt (see
`docs/perf.md`). `tools/pld_bench.v` measured the root cause: **accept rate
is 3/15 = 0.2 on the code-edit probe and does not change with max_ngram**
(2/3/5/8/16 all identical). At p=0.2 speculation cannot pay even with free
verification, so a different draft source — not a better kernel — would be
needed. The correctness foundation and accept-rate instrumentation are the
deliverables; no speedup is claimed. EAGLE3/DFlash/DSpark/MTP stay out.

* Ship only PLD/ngram in `src/infer/sampler.v` (pure V,
  arena-allocated, opt-in flag + `VLLM_*` env in `src/cfg/config.v`).
  Start from validated hyperparams: `max_ngram_size = 3`, continuation
  10, match over prompt + own output (turn-2 code editing is our best
  case). `EAGLE3/DFlash/DSpark`/built-in MTP stay out (need per-target
  draft weights or `blk.48`-style weights that do not fit the 4 GB guest).
* Gate: accept-rate + wall-time in `tools/forward_bench.v` on a code-edit
  prompt, greedy `temp: 0.0` bit-identical on/off, host-vs-guest parity
  like `infer_smoke.v`.

## Phase 9 — serve path (planned, 2026-11-19)

Goal: attack the measured prefill-bound serve wall (`chat8` ~1.5 s,
ChatML prompt ~1.2 s of it).

Sources: `EricLBuehler/mistral.rs` (prefix caching, chunked prefill,
ISQ, hardware-aware `tune`), modchip (THP/NTA/`/dev/shm` snapshot),
`intel/ipex-llm` mmap side-channel warning (ARCHIVED project — never a
dependency).

Status 2026-11-20: `--tune` landed (host CPU/RAM detection → backend/threads/
quant/ctx_cap, prints and exits). **Prefix caching landed** as an opt-in
(`model.prefix_cache` / `VLLM_PREFIX_CACHE=1`): one Mutex-protected entry
holding the last prompt's token ids and KV, copied into the next request's
ForwardCtx when its ids share that prefix; only the suffix is prefilled.
Bit-identical output is gated on real weights (skip when absent). Measured on
Qwen3-0.6B-Q4_0 with `VLLM_THREADS=1`: an 18-token shared prefix took
3063 ms uncached vs 972 ms cached (~3x); a 2-token prefix is slower
(copy overhead exceeds one prefill), so the cache is opt-in rather than
always-on. Chunked prefill, host-side ISQ and the tools-only THP/NTA
experiments remain; the security note on single-tenant mmap is in
`docs/security.md`.

Dispositions for the rest of Phase 9 (2026-11-20):

* **Chunked prefill — not applicable in v1.** Chunking only buys fairness
  when many requests share one prefill/decode scheduler. Here the accept
  loop spawns one handler thread per request (`server.max_conns` bounds
  concurrency), so a long prompt blocks only its own thread — other
  requests already proceed in parallel. Adding chunk boundaries without a
  shared scheduler cannot reorder work and would just add bookkeeping.
  Revisit only if the thread-per-request model is replaced.
* **Host-side ISQ — deferred.** There is no in-tree quantizer/llama.cpp
  conversion step to hook; `scripts/fetch_model.sh` downloads an already
  quantized GGUF. Adding an ISQ stage needs a reference converter and a
  real-weight quality gate first (the Phase 7 harness can supply the gate).
* **Tools-only THP / NTA / `/dev/shm` snapshot — deferred.** All three are
  Linux-specific (`madvise`, `_MM_HINT_NTA`, shm) and the current host is
  Windows with no quiet measurement window; the pool A/B was already
  BLOCKED on that basis. They belong in `tools/` behind a Linux CI lane
  where VmHWM and page-fault counts can be measured, not in the guest.

* Prefix caching (reuse prompt KV across turns) + chunked prefill
  (bound per-request prefill work) in the serve path; ISQ (in-situ
  quantization on load) host-side only, e.g. in the
  `scripts/fetch_model.sh` flow, never in the guest. Paged attention
  stays a research note (our flat arena in `src/infer/attention.v` is
  simpler and already request-sized).
* Hardware-aware `--tune` host flag (print recommended
  backend/threads/quant from detected CPU, like `mistralrs tune`).
* Memory/VM experiment (`tools/` only): `madvise(MADV_HUGEPAGE)`
  inward-aligned to 2 MiB for `Model.blob`, `_MM_HINT_NTA`-style
  streaming for weight sweeps vs L3-resident KV, host-side `/dev/shm`
  Slot-0 snapshot/restore (modchip: ~38 ms vs ~6 s prefill). No
  `mmap`/THP dependency in the freestanding path.
* Record in `docs/security.md`: our mmap'd `Model.blob` is safe only
  because the appliance is single-tenant by design.
* Gate per item: same greedy tokens, measured wall-time win on `chat8`,
  `vllm.example.json` + `docs/configuration.md` row for any new knob.

## Phase 10 — long memory (planned, 2026-11-19)

Goal: raise the 4k context ceiling without breaking the 4 GB guest.

Sources: modchip/Nemotron (6 GQA layers = 1.7 KB/token vs 27 KB/token
dense — the reason for our 4k cap), `LLAMA_ATTN_WINDOW`-style SWA;
docs-only boundaries: `Tiiny-AI/PowerInfer(-2)` (hot/cold needs
ReLU-sparse models; `Qwen3 SwiGLU` is dense — model-selection bias
toward MoE only), `kvcache-ai/KTransformers` (CPU+GPU offload needs
GPU drivers `vinix-mini` does not have — seam stays
`list_backends()`/`select_backend()`), `mlc-ai/mlc-llm` (TVM needs a
compiler runtime no freestanding guest can host — rejected with reason),
`microsoft/BitNet.cpp` (1.58-bit needs trained ternary weights; keep as
the sub-2bpw reference for the Phase 7 quality gates).

Status 2026-11-20: **landed**. `attention_step_window` attends to the union
of the first `attn_sinks` positions and the last `attn_window` positions,
deduplicated and allocation-free; `window <= 0` (the default) delegates to
exact full attention. Knobs `model.attn_window` / `model.attn_sinks` plus
`VLLM_ATTN_WINDOW` / `VLLM_ATTN_SINKS`, validation, and ForwardCtx wiring
are in place. Gates: exact fallback equality, agreement with a naive
index-list reference, and config defaults/env/validation. The
`tools/attn_bench.v` scaling probe measured full 121.6 us/head-step at
n=512 vs 1057.8 us at n=4096, and windowed 60.4 / 74.8 us (~2x / ~14x)
— recorded in `docs/perf.md`. The 4k default is unchanged until a >=16 GB
guest exists. Docs-only boundaries are written up in
`docs/architecture.md`.

* Code: opt-in sliding-window attention + sinks
  (`LLAMA_ATTN_WINDOW`-style knob in `src/cfg/config.v` +
  `src/infer/attention.v`, default exact full attention).
* Gate: attention-scaling probe (`n=512/4096` head-step) recorded in
  `docs/perf.md`; 4k default unchanged until a ≥16 GB guest exists.
  Everything else in this phase is docs-only (`docs/architecture.md`
  paragraph each), no guest code.

Non-goals (stay deferred): vendored `llama.cpp`/`ik_llama` diffs,
modchip custom types 44/45, draft-model GGUFs, GPU/NPU runtimes,
TVM/compiler toolchain in the image, training code, ternary weights,
ReLU-sparse-only models, distributed/RPC inference, multi-tenant mmap
exposure. Supersedes the Phase 7 A–E / Phase 8 F–J / Phase 9 K–O
itemization (same content, execution order).

---

## Phases 11–15 — from "works on 0.6B" to a usable appliance (2026-11-20)

The v1 goal was never fully met: the target weights are **Qwen3-1.7B-Q4_K_M**,
but the forward only handles Q4_0/Q4_1/Q6_K/IQ4_NL/Q2_0, so the target model
cannot run at all. Every earlier measurement used Qwen3-0.6B-Q4_0. These
phases close that gap first, then performance, then production-readiness.
Ordering is load-bearing: benchmarking an unsupported target is meaningless.

### Phase 11 — Q4_K_M target-model support (critical path, next up)

Why first: without it the v1 deliverable is non-functional. Block spec from
the GGML primary source (already fetched, no code copied):
`block_q4_K` = 256 elems / 142 bytes = f16 `d` + f16 `dmin` + `scales[12]` +
`qs[128]`; eight 32-elem sub-blocks with split-half nibbles; 6-bit
scale/min via `get_scale_min_k4`; `out = d*sc*q - dmin*m`.

* `gguf.v`: add `.q4_k` type and exact 142-byte / 256-elem sizing.
* `quant.v`: `qk_k=256`, `q4_k_block_bytes=142`, `BlockQ4K`, `block_q4_k_at`,
  `dequant_q4_k_row`, `q4_k_dot_row`, `q4_k_matvec`, `embed_gather_q4_k`.
* `vllm_kernels.{h,c}`: `vllm_q4_k_matvec_f32` — AVX2 row-vector accumulation
  with one horizontal sum per row (mirror the existing Q4 row-vector kernels),
  scalar fallback.
* `simd_c.v`, `parallel.v`, `worker_pool.v`, `forward.v`: bridge, range form,
  threaded + pooled head, `.q4_k` dispatch for embedding and vocab head.
* Gate: `q4_k_test.v` (code/dequant hand values, fused-vs-dequant, matvec
  reference, C-vs-V, threaded/pool, tail rows) + real 1.7B golden table +
  8-step greedy bit-identical to the reference and logit maxdiff < 1e-3.

### Phase 12 — close the ~3x gap to llama.cpp

* Proper SIMD-across-k matmul: k register accumulators, weight dequant shared
  across drafts, one hsum per row per token; finish the batched Q6_K head.
  Replace the current experimental `VLLM_PLD_BATCH=1` path.
* Draft source: accept rate is 3/15 = 0.2 and independent of `max_ngram`;
  literal-copy drafts diverge from the model's pattern generalisation. Try
  fuzzy matching / multi-match voting / shorter k. If the gate still fails,
  stop speculation and spend the time on the layout-level kernel work.
* Load-time row-interleaved repack behind `Backend`, only tensors the CPU
  really runs (the `ik -rtr` warning).
* Gate: 1.7B tok/s measured on a quiet box; no timing row without one.

### Phase 13 — real validation and CI

* QEMU 1.7B gate on the Q4_K_M model, real chat asserts, artifacts kept.
* Quiet Linux CI (VJOBS=2, no stray processes): run the three blocked A/Bs —
  pool on/off, PLD on/off, quant parity. Fill `docs/perf.md` with 1.7B rows.
* `quality_gates --max-rss-mb 4096` in CI (VmHWM ceiling).

### Phase 14 — production readiness

* SSE / cancellation / backpressure under load, request timeouts.
* Structured observability: request id, prompt tokens, decode ms, cache hit,
  accept rate in logs; `/health` with state.
* Memory budget checks before allocation (weights + KV + arena <= 4 GB).
* `--print-config` shows all new knobs; security log-hygiene review.

### Phase 15 — long context and remaining experiments

* SWA + sinks on a >=16 GB guest: PPL/VmHWM probe, then decide default.
* Multi-entry LRU prefix cache with extend-on-hit (current single-entry cache
  never grows on a hit).
* Chunked prefill / host ISQ / THP/NTA stay dispositioned until the
  thread-per-request model or a Linux measurement lane changes.

### Gates summary

| phase | exit gate | blocked by |
|---|---|---|
| 11 | 1.7B Q4_K_M greedy bit-identical + golden table | — |
| 12 | 1.7B tok/s row on a quiet box | 11 |
| 13 | three A/Bs green in CI | 12 + quiet host |
| 14 | SSE/cancel/observability tested | 13 |
| 15 | SWA gate on >=16 GB guest | 13 |

## Deferred (deliberately out of v1)

GPU / NPU drivers, multi-tenant auth + rate limits, full 32k ctx,
quant servers beyond IQ4_NL/Q2_0 (Phase 7), ARM64 bare metal, OTA updates.
