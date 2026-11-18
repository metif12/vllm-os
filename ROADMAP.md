# vllm-os roadmap — Qwen3-1.7B on trimmed Vinix

Each phase has an exit gate. Do not start the next phase until the gate is green.

## Phase 0 — Host skeleton (done ✅)

`v run src` serves OpenAI-compat stub on Windows/Linux.
Gate: `/v1/models` + `/v1/chat/completions` return 200. Green.

## Phase 1 — Host inference (core done ✅, forward pass = 1b ⏳)

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

Follow-ups (not gates): server log/API still print config-default
`Qwen/Qwen3-1.7B` / `Q4_K_M` labels instead of GGUF-derived id/quant;
Llama-3.2-1B golden fixture not yet fetched (773 MB, optional).

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

## Deferred (deliberately out of v1)

GPU / NPU drivers, multi-tenant auth + rate limits, full 32k ctx,
quant servers beyond Q4_K_M/Q8_0, ARM64 bare metal, OTA updates.
