# vllm-os image build (Phase 5). Runs under WSL2 Ubuntu / Debian Linux.
#
#   make image [MODEL=...] [TOKENIZER=...] [CONFIG=...] [V=...]
#              [KERNEL=...] [OUT=...] [WORK=...] [EXTRA=...]
#
# Assembles vllm-os.iso: the vinix-mini kernel, a static /sbin/vllm
# userland server, the baked weights + tokenizer + config.json, and the
# Limine BIOS+UEFI boot layout (see kernel/init/build-image.sh).
# Boot it with: qemu-system-x86_64 -cdrom vllm-os.iso -m 4G
#   -display none -serial stdio -netdev user,id=n0,hostfwd=tcp::18000-:8000
#   -device e1000,netdev=n0   (or scripts/mini_boot_gate.ps1 on Windows)
V ?= $(shell command -v v 2>/dev/null || echo $(HOME)/vlinux/v/v)
MODEL ?= models/Qwen3-0.6B-Q4_0.gguf
TOKENIZER ?= $(dir $(MODEL))tokenizer.json
CONFIG ?=
# KERNEL must point at a built vinix-mini binary: either build the fork
# in-tree (kernel/mini/bin/vinix, see kernel/BASELINE.md for the pinned
# toolchain) or pass KERNEL=/path/to/bin/vinix explicitly.
KERNEL ?= kernel/mini/bin/vinix
OUT ?= vllm-os.iso
WORK ?= build/image-work
EXTRA ?= build/image-extra

SRC_V := $(shell find src tools kernel/init \( -name '*.v' -o -name '*.c' -o -name '*.h' \) | sort)

.PHONY: image clean-image

image: $(OUT)

$(OUT): $(KERNEL) $(MODEL) $(TOKENIZER) $(SRC_V) kernel/init/build-image.sh GNUmakefile
	mkdir -p $(EXTRA)/sbin $(EXTRA)/models
	$(V) -prod -cc gcc -path "src|@vlib|@vmodules" -cflags '-static' -o $(EXTRA)/sbin/vllm src
	chmod 755 $(EXTRA)/sbin/vllm
	cp $(MODEL) $(EXTRA)/models/model.gguf
	cp $(TOKENIZER) $(EXTRA)/models/tokenizer.json
	sh kernel/init/build-image.sh --kernel $(KERNEL) --out $(OUT) --work $(WORK) \
	  --initramfs-extra $(EXTRA) $(if $(CONFIG),--config $(CONFIG))

clean-image:
	rm -rf $(WORK) $(EXTRA) $(OUT)
