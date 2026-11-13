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
| `host-native` | `v -prod -cc gcc -cflags "-O3 -fno-strict-aliasing -flto" -o server src` | ~1.20 MB | LTO cross-module inlining; within noise of release, smaller |
| `host-minimal` | `v -cc gcc -cflags "-Os" -o server src` | ~1.11 MB | no C layer, asserts on; auditability over speed (~+4 ms p50) |
| `qemu` | release build, then guest test | — | `run ./scripts/qemu_test.ps1 next` |
| `bare-metal` | `v -prod -gc none -freestanding -cc gcc -o kernel/vinix-mini.elf src` | elf | experimental freestanding kernel-port |

```sh
sh scripts/build.sh host-release  # linux/mac (mirrors the table)
```

Custom `-cflags` REPLACE V's default flag set: any profile adding
flags must restate `-O3 -fno-strict-aliasing` (bare `-O3` measured
~2x slower). Never `-march=native` in committed scripts: it
pessimize V's own codegen 3-5x (measured) AND the ISO must boot
anywhere. SIMD does not need it — see `docs/perf.md` flag hygiene.

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

```sh
sh kernel/fetch_vinix.sh        # pin upstream Vinix into third_party/
sh scripts/build_image.sh [weights.gguf]
```

`build_image.sh` validates the weights + `kernel/vinix-mini.elf`, stages
kernel (stripped) + `models/model.gguf` + `tokenizer.json` + `limine.cfg`
(`TIMEOUT=0`, serial console, `model=/models/model.gguf …`), and emits
`vllm-os.iso` via xorriso. Feature selection lives in
`kernel/features.conf` (every extra OFF by default). Link constraints
in `kernel/mini.ld` (higher-half, 2 MB-aligned, debug sections
discarded); boot order, IP defaults, and the memory budget in
`kernel/mini.md`.
