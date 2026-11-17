# qemu/ — test harness

Phase-1/2 test does NOT need the vinix-mini ISO yet. It proves the
guest→host network path that the later bootable image will rely on.

## Prereqs

* Host server built (`scripts/qemu_test.*` builds it if missing).
  Build with the **release** toolchain only (see `docs/development.md` —
  V 0.5.2 release `7647ce1`): `v -prod -cc gcc -o server.exe src`.
* QEMU only needed for the guest leg:
  `winget install -e --id SoftwareFreedomConservancy.QEMU` (Windows)
  or your distro package (Linux).

## What `scripts/qemu_test.ps1` / `qemu_test.sh` do

1. Start `server[.exe] --port 8000` on the host (stub mode OK, no weights).
2. Wait for the port (poll, no fixed sleep) and record boot time.
3. Assert `GET /v1/models` → 200 + id `qwen3-1.7b`.
4. Assert `POST /v1/chat/completions` → 200 + non-empty content.
5. Gate: **server start → first token < 30 s wall** (stub).
6. Guest leg (only when `GUEST_IMG` points at a guest image): boot the
   microvm on SLIRP user networking and prove the guest reaches the host
   server at `10.0.2.2:8000` (SLIRP gateway — `hostfwd` is never used for
   the tested direction, only for the optional ssh channel).
   * With `GUEST_SSH_PORT` + `GUEST_SSH_KEY`: ssh into the guest and curl
     `http://10.0.2.2:$Port/v1/models` from inside.
   * Default (no ssh): serial-console automation. The guest boots with a
     TCP serial; the script drives the **OpenWrt failsafe shell** over it
     (normal boot stalls pre-console on some QEMU/host combos — failsafe
     does not), configures `10.0.2.15/24` + default route, then asserts
     `/v1/models` **and** a full `/v1/chat/completions` round trip from
     inside the guest (this caught real bugs: uclient-fetch sends
     `Transfer-Encoding: chunked`, which the server now decodes).

Tested guest: OpenWrt 25.12.6 x86-64 `ext4-combined.img` (raw, `if=virtio`):

```powershell
$env:GUEST_IMG = "$PWD\qemu\openwrt.img"
.\scripts\qemu_test.ps1 -Port 8000   # GUEST-GATE PASS via serial failsafe
```

Knobs: `GUEST_SERIAL_PORT` (default 4446), `QEMU_EXTRA_ARGS`
(default `-accel whpx -machine q35` on Windows; use
`-accel kvm -machine q35` or empty on Linux), `TIMEOUT_SECS` (host
readiness deadline). Guest images (`*.img`/`*.iso`), `guest.log`,
`qemu.out/.err`, `*.pid` under `qemu/` are gitignored.

Exit codes: `0` host PASS (guest PASS/SKIP noted), `1` host FAIL,
`2` prereq/port issue, `3` guest attempted but FAILED.

## Run

```powershell
.\scripts\qemu_test.ps1 -Port 8000
$env:GUEST_IMG = 'C:\images\alpine.qcow2'   # optional guest leg
.\scripts\qemu_test.ps1
```

```sh
./scripts/qemu_test.sh 8000
GUEST_IMG=~/images/alpine.qcow2 ./scripts/qemu_test.sh
```

Logs: server output → `qemu/test.log`, guest serial → `qemu/guest.log`
(both gitignored). If `:8000` is already serving, the scripts assert
against it read-only and exclude boot time (noted in output).

## Phase-3 upgrade

Once `kernel/` boots, the guest leg switches to
`qemu-system-x86_64 -cdrom vllm-os.iso -m 4G -display none -serial stdio`
and waits for `[vllm-os] boot ok` instead of a Linux guest image.
