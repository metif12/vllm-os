# Building & running

## Profiles

```powershell
.\scripts\build.ps1 -Profile host-release   # windows daily driver
```

```sh
sh scripts/build.sh host-debug     # fast iteration: v -o server src
sh scripts/build.sh host-release   # what you ship on host: v -prod -o server src
sh scripts/build.sh qemu           # release build, then run the guest test
sh scripts/build.sh bare-metal     # experimental: -gc none -freestanding kernel elf
```

`-prod` enables `-O3` and strips asserts. Never commit `-march=native`:
the ISO must boot anywhere.

## Host run (no weights needed)

```sh
v run src --model models/qwen3-1.7b-q4_k_m.gguf --port 8000
```

Missing weights file → stub mode: the API is live, generations are
clearly marked. Good for HTTP/client work before downloading gigabytes.

## QEMU network-path test

Needs `qemu-system-x86_64` on PATH
(`winget install -e --id SoftwareFreedomConservancy.QEMU`).

```powershell
v -prod -o server.exe src
.\scripts\qemu_test.ps1 [-Port 8000]
```

Phase 1: host loopback asserts (`/v1/models` id, chat 200 + text).
Phase 2: set `$env:GUEST_IMG` to a qcow2 image and the harness boots it
with SLIRP `hostfwd` so the guest reaches the host at `10.0.2.2:<port>`
— the same path the bootable ISO will use. Full guest-image flow is in
`qemu/README.md`.

## OS image (phases 3–5)

```sh
sh kernel/fetch_vinix.sh        # pin upstream Vinix into third_party/
sh scripts/build_image.sh [weights.gguf]
```

`build_image.sh` validates the weights + `kernel/vinix-mini.elf`, stages
kernel (stripped) + `models/model.gguf` + `tokenizer.json` + `limine.cfg`
(`TIMEOUT=0`, serial console, `model=/models/model.gguf …`), and emits
`vllm-os.iso` via xorriso. Link constraints live in `kernel/mini.ld`
(higher-half, 2 MB-aligned, debug sections discarded); boot order, IP
defaults, and the memory budget live in `kernel/mini.md`.
