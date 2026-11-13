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
* ⏳ Phase 1b: transformer forward (attention + KV-cache) for real logits;
  pseudo-logit path exercises encode→sample→decode end to end until then.
* Gate: perplexity smoke test vs `llama.cpp` on 3 prompts (needs 1b + weights).

## Phase 2 — QEMU network path

* Host server + QEMU guest curls it via SLIRP (`10.0.2.2:8000`).
* `scripts/qemu_test.ps1/.sh` automated: boot microvm, curl models +
  chat, assert 200 + `qwen3-1.7b` id.
* Gate: cold-boot guest → first token < 30 s wall (stub), log saved.

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

## Phase 4 — Bare-metal inference

* Build `src/infer` with `-gc none -freestanding`, arena KV-cache.
* mmap GGUF from baked partition (no `os` file APIs in kernel).
* Single accept loop + N worker threads on the trimmed scheduler.
* Gate: QEMU `-m 4G` serves a real (non-stub) completion from the ISO.

## Phase 5 — OS+weights image

* `make image MODEL=models/qwen3-1.7b-q4_k_m.gguf` → `vllm-os.iso`.
* Contents: kernel + `models/*.gguf` + `tokenizer.json` + `config.json`.
* Outside world sees only `:8000` OpenAI API. No SSH, no shell.
* Gate: fresh QEMU boot from ISO, no host files, chat completion works.

## Deferred (deliberately out of v1)

GPU / NPU drivers, multi-tenant auth + rate limits, full 32k ctx,
quant servers beyond Q4_K_M/Q8_0, ARM64 bare metal, OTA updates.
