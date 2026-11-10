# qemu/ — test harness

Phase-1/2 test does NOT need the vinix-mini ISO yet. It proves the
guest→host network path that the later bootable image will rely on.

## Prereqs

* `qemu-system-x86_64` on PATH.
  Windows: `winget install -e --id SoftwareFreedomConservancy.QEMU`
* Host server built: `v -o server.exe src` then running on `:8000`.

## What `scripts/qemu_test.ps1` does

1. Starts `server.exe --port 8000` on the host (stub mode OK).
2. Boots a tiny QEMU microvm (SLIRP user networking).
3. Inside the guest (or via host-side SLIRP forward), curls
   `http://10.0.2.2:8000/v1/models` and `/v1/chat/completions`.
4. Asserts HTTP 200 + model id `qwen3-1.7b`, saves the log.

## Run

```powershell
v -o server.exe src
.\scripts\qemu_test.ps1 -Port 8000
```

## Phase-3 upgrade

Once `kernel/` boots, this same script switches to
`qemu-system-x86_64 -cdrom vllm-os.iso -m 4G -display none -serial stdio`
and waits for `[vllm-os] boot ok` instead of using a Linux guest image.
