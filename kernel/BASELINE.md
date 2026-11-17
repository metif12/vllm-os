# kernel/BASELINE — stock Vinix build status (Phase 3 step 1)

Upstream: `vlang/vinix` @ `63fc675f` (2026-11-14, recorded in
`kernel/upstream.pin`, cloned to `third_party/vinix`, also mirrored at
`~/vinix-baseline` in WSL Ubuntu for builds).

## Toolchain (verified working, WSL2 Ubuntu, no sudo needed)

* Kernel V: built from `tools/m1-wifi/get-v.sh` pin `7490a8ff`
  (`V 0.5.2 7490a8f`), `HOST_CC=gcc` — clang is NOT required despite the
  CI default. Lives at `~/vkernel` (outside the repo).
* Utility V: 0.5.2 release (`7647ce1`, same pin as this repo).
* Deps: `kernel/get-deps` (limine, lwip, …) over proxy — green.
* Docker path abandoned: daemon-side pulls stall; WSL-native is faster.

## Upstream is RED at this pin (not our environment)

* `vlang/vinix` "Build Vinix kernel": 12 consecutive failures Nov 11–17,
  including the push of this exact pin (`43568965864`) and every nightly
  since. No green kernel build in the window.
* Upstream fails in the same step we fail in ("Attempt to build the
  Vinix kernel (debug)", ~7 s in = V semantic check, not codegen).
* Our local failure mode with the pinned V: `C.memcpy` / `C.memset` /
  `C.__builtin_alloca` re-declared with clashing signatures across
  modules (e.g. `term/sleep.v` vs `socket/inet/net_random.v`,
  `initramfs/gzip.v` vs `socket/inet`, `dev/ath10k` vs `userland`,
  `block/verity` vs `fs`). Same class our repo already bans in
  `docs/development.md` (no module-level `fn C.*` re-decls; use vlib's).

## Implication for vinix-mini

* There is no green upstream baseline to fork from. Do not wait for one.
* The trim deletes some offenders outright (`dev/ath10k`, `dev/e1000`,
  HID/audio, userland — see `kernel/mini.md` drop list).
* Offenders in KEPT modules (`fs`, `socket/inet`, `term`, `initramfs`)
  must be unified to vlib declarations as part of the fork — budget for
  it, it is not free.
* Next: copy `third_party/vinix/kernel` → `kernel/mini`, apply
  `features.conf` deletions, fix the remaining C-decl clashes, build
  with the `~/vkernel` V + gcc, boot to `[vllm-os] boot ok` under QEMU.

## V rule found (2026-11-17, verified by experiment)

`uint` is NOT a V builtin in the pinned compiler (`7490a8f`) — in C or V
declarations it resolves to a per-module phantom type (`inet__uint`,
`netcore__uint`, ...). Consequences:

* Any `fn C.*` using `uint` params clashes ACROSS modules ("already
  declared with a different signature"), even when byte-identical.
  Minimal repro: two modules each declaring
  `fn C.__builtin_alloca(uint) voidptr` fail; with `u32` they pass.
* V semantic check passes these through, then C compile fails with
  `unknown type name 'mod__uint'`.

Fix applied in `kernel/mini` (10 files, decl lines only): `uint` → `u32`
(term/sleep, netcore, initramfs/gzip, block/verity, aesxts, fs/ext2,
dev/ath10k, socket/inet, fs/nfs, userland/linux_task). After this the V
check is fully clean; C compile then reports the remaining `uint` uses
(~103, in lib, netcore, inet, stubs, krandom, fs, kprint + dropped
modules) — same one-word mechanical fix, module by module.
Call sites passing `u32` values are unaffected; `uint(...)` casts
(`lib/stack_protector.v`, verity) become `u32(...)`.

## Phase 3 outcome (2026-11-17): fork builds, boots to init handoff

`kernel/mini` compiles with zero C errors and links `bin/vinix`;
a Limine/BIOS+UEFI ISO (xorriso + pinned Limine 12.9.3, both fetched
without sudo) boots under QEMU `-m 2G` through ACPI/SMP/scheduler,
RNG, lwIP loopback + e1000/DHCP, all mounts and initramfs unpack, and
stops at the expected `V panic: Could not start init process`
(no userland yet). Serial log is the evidence (kept outside the repo).

Further pinned-V findings (each fixed in `kernel/mini`, marked
`NOTE(vllm-os)` in-tree):

* `@type` on C structs mangles to `._v_type` in generated C (V structs
  are consistent, C structs are not). Fixed with C accessors
  (`c/vinix_ip_type.{h,c}` wrapping lwIP `IP_GET_TYPE`/`IP_SET_TYPE`),
  ~60 use sites in `netcore/core.v` rewritten by script.
* `&` on a function-pointer value is dropped (`memcpy(&n.output,
  &callback, …)` becomes `memcpy(n->output, callback, …)` — copies
  callee code to the old pointer, then faults writing read-only
  `.text`). Fixed with `static inline` C store helpers taking `void*`
  (`c/net_driver_abi.h`).
* `u32(pointer)` truncates high-half addresses (stack/heap/direct map)
  → page faults. Fixed to `u64` wherever the value feeds address
  arithmetic (`krandom/erase.v`, `aesxts`, `abiargs`, `initramfs`,
  `netcore` pbuf copy, `net_random` cursor, `printf_entries` varargs).
  Pure length/count/alignment/flag uses of `u32` are unaffected.
* `x is <interface>` with ZERO implementors in the program compiles to
  a non-nil check (always true) and the subsequent dispatch panics
  (upstream documents the same trap for `is MetadataResource` in
  `fs/tmpfs.v`). Forced-false with rationale comments at every site:
  `MountSpecifier` (mount path), `RenamePublication`,
  `FallibleFileSystemStatResource`, `MappingAttributesResource`,
  `StatSnapshotResource`, `MetadataUpdateResource`,
  `PrivateMountOptions`. Interfaces with ≥1 implementor generate
  correct typ-id chains (verified in blob.c).
* `name.str` on a map-iteration key miscompiles to a closure call
  (breaks the `-no-closures` build); `hibernate/machine_amd64.v` now
  hashes an explicit byte copy. A two-element fixed-array literal in
  `sysvsem` referenced an undeclared spill; expanded to indexed stores.
* `sizeof(global)` on a V `__global` emits the bare name
  (`sizeof(net_physical_netif)`); replaced with the concrete C type.
* Heap-bench sampler dropped from the fork makefile: its host
  generator tool needs newer-V spread syntax the pinned V rejects.
  The sampler is opt-in (`-d heap_benchmark`) and unused by mini.
