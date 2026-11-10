# vllm-os roadmap — Qwen3-1.7B on trimmed Vinix

Each phase has an exit gate. Do not start the next phase until the gate is green.

## Phase 0 — Host skeleton (now)

Goal: `v run src` serves OpenAI-compat stub on Windows/Linux.
Gate: `curl /v1/models` + `curl /v1/chat/completions` return 200,
`v fmt -w` clean, `v vet` clean.

## Phase 1 — Host inference (real tokens)

* Port GGUF tensor table + KV metadata (beyond the 24-byte header).
* Pure-V kernels: dequant (Q4_0/Q4_K/Q8_0), RMSNorm, SwiGLU, RoPE,
  GQA attention + KV-cache, sampler with RNG.
* Qwen BPE tokenizer port (151k vocab) + `tokenizer.json` loader.
* CLI: `server --model x.gguf --prompt 'hi' --max-tokens 64` prints tokens.
* Target: Qwen3-1.7B-Q4_K_M, ≥5 tok/s on a laptop CPU, 4k ctx.
* Gate: perplexity smoke test vs `llama.cpp` reference on 3 prompts.

## Phase 2 — QEMU network path

* Host server + QEMU guest curls it via SLIRP (`10.0.2.2:8000`).
* `scripts/qemu_test.ps1/.sh` automated: boot microvm, curl models +
  chat, assert 200 + `qwen3-1.7b` id.
* Gate: cold-boot guest → first token < 30 s wall (stub), log saved.

## Phase 3 — vinix-mini fork

Fork Vinix, keep only:

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
