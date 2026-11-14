# kernel/mini — vinix-mini trim + boot optimization spec

Concrete keep/drop list with size and boot-time impact. All choices serve
one goal: power-on → first token as fast as possible, RAM left for weights.

Compile-time selection lives in `kernel/features.conf`: every extra is a
`VLLM_MINI_*` flag defaulting to OFF, and fallback drivers (NVMe/AHCI,
e1000, ramfs) are deleted from the fork, not flagged. An image built with
all flags at default is the minimal boot-to-token set below.

## Boot sequence (optimized order)

1. `serial init` first — every later stage logs timestamps here.
2. `pmm/vmm/kheap` with 2 MB + 1 GB huge pages for the weights window.
3. `virtio-blk` only (AHCI/NVMe fallback deleted from the fork:
   saves ~200 ms + driver pages). Single queue, read-only mount of the
   models partition (tarfs; ramfs deleted — see `features.conf`).
4. `mmap` GGUF via page-table alias, parse KV + tensor table only.
   Weight blobs stay on disk until first matmul touches the page.
5. `virtio-net` only (e1000 fallback deleted), static IP from kernel
   cmdline (`ip=... gw=...`, `dhcp=off`), no DHCP wait (saves 1–3 s worst
   case). DHCP + UDP stack compile back in only with
   `VLLM_MINI_NET_DHCP=1` and `dhcp=on` on the cmdline.
6. Scheduler up with `workers = cpus - 1`; accept loop pinned to CPU 0.
7. Listen `:8000`, print `[vllm-os] boot ok in ${ms} ms`.

## Dropped (with savings)

| dropped | saves |
|---|---|
| framebuffer/DE/fonts | ~8–15 MB image, mode-set time |
| e1000 + AHCI/NVMe probe | ~200–400 ms boot, ~100 KB text |
| USB HID/audio | drivers + IRQ threads |
| DHCP client (static default, DHCP opt-in) | up to 3 s boot |
| shell/coreutils/userland | whole userland image |
| syscalls beyond socket/mmap/clock | vDSO surface, audit time |
| file write path (models RO) | no journal, no fsync stalls |

## Memory budget (4 GB guest, Qwen3-1.7B Q4_K_M)

| region | size |
|---|---|
| kernel text/rodata | ~4 MB |
| page tables + per-CPU stacks (8 CPUs) | ~32 MB |
| weights mmap window (1.1 GB GGUF) | 1126 MB (demand-paged) |
| KV-cache arena, 4k ctx | ~256 MB cap |
| net buffers (256 x 4 KB rings) | 1 MB |
| request scratch (per worker) | 8 x 2 MB |
| headroom | ~2.4 GB free for page cache |

Context cap 4k is the load-bearing trim: full 32k KV would exceed the
budget by ~2 GB. Raise only on ≥16 GB guests.

## Runner build flags (host + kernel port)

All builds use an optimizing C compiler (`-cc gcc`) in `-prod`
(`-O3`, asserts stripped). tcc is never used: scalar C fallbacks exist
so every profile compiles anywhere, and AVX2 comes from function-target
attributes + runtime dispatch — never `-march=native`, which gcc's SLP
vectorizer turns into a shuffle swamp ~2.4x slower than scalar on V's
emitted loops (minimal repro + asm in `docs/perf.md`).

* Host release: `v -prod -cc gcc -o server src` (portable AND fastest:
  AVX2-dispatched C kernels, best V codegen; custom `-cflags` append to
  defaults, and extra flags measured identical — no "native" profile needed).
* Host minimal (no C layer): `v -cc gcc -cflags "-Os" -o server src`.
* Kernel port adds: `-gc none -freestanding`, arena allocator, no `os`/`time` wall-clock in token loop.
* Never `-march=native` in committed scripts: the ISO must boot anywhere.

## Measurable gates

* `qemu -m 4G -display none -serial stdio`: `boot ok` line < 2 s after SeaBIOS.
* First `/v1/chat/completions` byte < 5 s after boot line (stub weights).
* `v test src` stays green (10 files), including the `[bench]` gates
  (`sample_into` throughput floor, blocked-vs-naive matvec equivalence + speed).
* Server holds max 64 concurrent connections (`max_conns` semaphore);
  excess load waits at accept instead of spawning unbounded threads.
