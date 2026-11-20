# Building & running

All profiles use gcc (`-cc gcc`); tcc is never used. `-prod` links the
C SIMD layer (`src/infer/vllm_kernels.c`, AVX2 via function attributes +
runtime dispatch) and strips asserts. Non-prod builds are pure V.

## Profiles

```powershell
.\scripts\build.ps1 -Profile host-release   # windows daily driver
```

| profile | command | binary | when |
|---|---|---|---|
| `host-debug` | `v -cc gcc -g -o server src` | ~1.1 MB + asserts | fast iteration; no C layer linked |
| `host-release` | `v -prod -cc gcc -o server src` | ~1.27 MB | default: portable AND fastest (SIMD dispatched) |
| `host-minimal` | `v -cc gcc -cflags "-Os" -o server src` | ~1.11 MB | no C layer, asserts on; auditability over speed (~+4 ms p50) |
| `qemu` | release build, then guest test | — | `run ./scripts/qemu_test.ps1 next` |
| `bare-metal` | `v -prod -gc none -freestanding -cc gcc -o kernel/vinix-mini.elf src` | elf | experimental freestanding kernel-port |

```sh
sh scripts/build.sh host-release  # linux/mac (mirrors the table)
```

Custom `-cflags` APPEND to V's defaults (verified via `-dump-c-flags`
+ identical binaries for `-O3` / `-O3 -fno-strict-aliasing` / no extra
flags), so plain `-prod` is already optimal and a separate "native"
profile was retired as redundant. Never `-march=native` on V code in
committed scripts: gcc's SLP vectorizer turns the 4-accumulator matvec
into a vpermd/vblendps swamp ~2.4x slower than scalar (minimal repro +
asm in the vlang/v issue), AND the ISO must boot anywhere. SIMD comes
from function-target attributes + runtime dispatch — see `docs/perf.md`
flag hygiene.

## Real-weight fixtures (golden gates)

The GGUF fixtures are not committed and must be fetched on the host:

```sh
sh scripts/fetch_small_models.sh          # default: Q4_0 fixtures (0.6B + 1.7B)
sh scripts/fetch_small_models.sh q4_k_m  # Phase 11 target only
sh scripts/fetch_small_models.sh all     # both sets
```

`q4_k_m` downloads `models/Qwen3-1.7B-Q4_K_M.gguf` (exact size asserted:
1107409472 bytes). It is needed for the Phase 11 real-weight gate —
`test_gguf_golden_qwen_17b_q4_k_m` in `src/infer/gguf_golden_test.v`
skips with `[skip] ... phase 11 real-weight gate pending` until that
file is present. Fetches are size-asserted and skip files already present
at the expected size; `http_proxy`/`https_proxy` are honoured by curl.
Note the golden test resolves the exact case-sensitive filename
`Qwen3-1.7B-Q4_K_M.gguf` (or set `VLLM_MODELS` to its directory).

## Host run (no weights needed)

```sh
v -prod -cc gcc run src --model models/qwen3-1.7b-q4_k_m.gguf --port 8000
```

Missing weights file -> stub mode: the API is live, generations are
clearly marked. Good for HTTP/client work before downloading gigabytes.

Boot log shows the resolved compute path, e.g.
`backend=cpu resolved=cpu simd=true`. `VLLM_NO_SIMD=1` forces pure-V
kernels at runtime (per-machine override, no rebuild); `backend` in
`vllm.json` selects `cpu | cpu-blocked | cpu-simd | auto`.

## QEMU network-path test

Needs `qemu-system-x86_64` on PATH
(`winget install -e --id SoftwareFreedomConservancy.QEMU`).

```powershell
v -prod -cc gcc -o server.exe src
.\scripts\qemu_test.ps1 [-Port 8000]
```

Phase 1: host loopback asserts (`/v1/models` id, chat 200 + text).
Phase 2: set `$env:GUEST_IMG` to a qcow2 image and the harness boots it
with SLIRP `hostfwd` so the guest reaches the host at `10.0.2.2:<port>`
— the same path the bootable ISO will use. Full guest-image flow is in
`qemu/README.md`.

## OS image (phases 3-5)

One-time: `sh kernel/fetch_vinix.sh` pins upstream Vinix into
`third_party/` (the `kernel/mini` fork derives from it).

```sh
make image                        # WSL: static server + 0.6B weights + config -> vllm-os.iso
make image MODEL=models/foo.gguf CONFIG=custom.json   # other weights / explicit config
```

`make image` (WSL2 Ubuntu, no sudo) builds the static `/sbin/vllm`
userland server, stages `sbin/vllm` + `models/model.gguf` +
`models/tokenizer.json` + `models/config.json` (explicit `--config`
file, else a minimal baked default pointing at `/models/model.gguf`),
and emits Limine BIOS+UEFI `vllm-os.iso` via
`kernel/init/build-image.sh` (pinned limine-binary 12.9.3, xorriso
bootstrapped without sudo; honours `http_proxy`/`https_proxy` and
`VLLM_CACHE`). Outside world sees only `:8000`. Boot it with
`qemu-system-x86_64 -cdrom vllm-os.iso -m 4G -display none -serial
stdio` (+ SLIRP hostfwd, see `scripts/mini_boot_gate.ps1`).

## ISO builder (phase 6: custom models)

```sh
python3 scripts/build_iso.py --model path/to/foo.gguf   # interactive knobs
python3 scripts/build_iso.py --model m.gguf --port 8080 --ctx 2048 \
  --preset precise --temp 0.5 --auth-token s3cret --dry-run   # print, don't build
```

Validates arch (x86_64 only — arm64 refused, fork drivers are
amd64-only), weights size vs guest RAM budget, tokenizer sidecar,
port/ctx/backend/preset/temp ranges — then generates the baked
`config.json` from knobs (or bakes `--config` as-is) and drives
`make image` (`MODEL` / `TOKENIZER` / `CONFIG` / `KERNEL` passthrough).
`gui/` is the same builder as a vlang/ui2 desktop app (pure
`gui/compose` logic, unit-tested; thin ui2 shell; builds with a
json2-capable V, see `docs/development.md` toolchain exception).
