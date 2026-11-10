# kernel/ — vinix-mini fork plan

Upstream Vinix is a full hobby OS. The fork keeps the minimum that can
boot, read weights off disk, and serve TCP on :8000.

## Keep

`boot / limine / GDT / IDT / paging / pmm / vmm / kheap / SMP bringup /
scheduler / spinlock+mutex / virtio-blk (primary) + NVMe fallback /
virtio-net (primary) + e1000 fallback / TCP+UDP+IP+DHCP / sockets /
tarfs+ramfs / ELF loader / serial logging.`

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
