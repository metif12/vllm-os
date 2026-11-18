# kernel/ — vinix-mini fork plan

Upstream Vinix is a full hobby OS. The fork keeps the minimum that can
boot, read weights off disk, and serve TCP on :8000.

## Keep (minimal boot-to-token set — see `features.conf`)

`boot / limine / GDT / IDT / paging / pmm / vmm / kheap / SMP bringup /
scheduler / spinlock+mutex / virtio-blk (only disk driver) /
virtio-net (only NIC driver, static IP) / TCP+IP (UDP only with DHCP) /
sockets / tarfs read-only / ELF loader / serial logging.`

Everything not listed here is OUT of the fork. NVMe/AHCI and e1000
fallbacks, ramfs, the DHCP client (opt-in via `VLLM_MINI_NET_DHCP=1` +
`dhcp=on` cmdline), USB HID, audio, framebuffer shell, pipes/signals,
extra syscalls, the writable-root path, and all of userland were removed
— not stubbed, deleted. `kernel/features.conf` is the single source of
truth; each extra is a flag defaulting to OFF.

## Drop

Framebuffer shell, desktop/DE, USB HID, audio, most POSIX syscalls,
pipes/signals, userland shell and coreutils.

## Why the scheduler stays

Inference server = 1 accept loop + N decode workers. A cooperative
"run-to-completion" stub deadlocks under concurrent SSE streams, so the
preemptive scheduler + mutexes are load-bearing and NOT trimmed.

## Next step (phase 3)

1. Submodule or subtree upstream Vinix at a pinned commit.
2. Feature-flag `VLLM_MINI` to exclude dropped drivers at compile time.
3. First gate binary: kernel that prints boot line over serial and
   answers TCP SYN — no inference yet.

## Reproducing the fork build (done in CI, see model-17b.yml)

`kernel/mini/get-deps` is upstream's script verbatim (byte-identical to
`vlang/vinix` at `kernel/upstream.pin`) — it vendors the C third-party
tree (freestnd headers, cc-runtime, flanterm, nanoprintf, uACPI, lwIP)
that `.gitignore` keeps out of the repo. With those outputs in place,
the fork builds standalone with the pinned kernel V
(`~/vkernel`, `7490a8f`, `HOST_CC=gcc`): `make -C kernel/mini
V=<vkernel-v>`. No upstream checkout needed (all test-only
`../tests`/`../build-support` references sit behind flags that default
OFF).
