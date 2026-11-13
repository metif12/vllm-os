#!/usr/bin/env sh
# Phase 5: assemble the OS+weights boot image. Host-only tooling.
# Usage: sh scripts/build_image.sh [path/to/qwen3-1.7b-q4_k_m.gguf]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF="${1:-$ROOT/models/qwen3-1.7b-q4_k_m.gguf}"
OUT="$ROOT/vllm-os.iso"
if [ ! -f "$GGUF" ]; then
  echo "weights not found: $GGUF (run scripts/fetch_model.sh first)" >&2
  exit 1
fi
if [ ! -f "$ROOT/kernel/vinix-mini.elf" ]; then
  echo 'kernel not built yet: phase 3/4 must produce kernel/vinix-mini.elf first' >&2
  exit 2
fi
STAGE="$ROOT/.imgstage"
rm -rf "$STAGE"
mkdir -p "$STAGE/boot" "$STAGE/models"
# features.conf <-> cmdline.txt consistency: a static-IP image must not
# carry dhcp=on (and vice versa). The fork build asserts the same flags.
FEAT="$ROOT/kernel/features.conf"
CMDLINE="$ROOT/kernel/cmdline.txt"
if [ -f "$FEAT" ]; then
  if grep -q '^VLLM_MINI_NET_DHCP=1' "$FEAT"; then
    grep -q '^dhcp=on' "$CMDLINE" || echo 'warn: features.conf enables DHCP but cmdline.txt has no dhcp=on'
  else
    if grep -q '^dhcp=on' "$CMDLINE"; then
      echo 'warn: cmdline.txt sets dhcp=on but features.conf leaves VLLM_MINI_NET_DHCP=0 (no DHCP client in image)' >&2
    fi
  fi
fi
cp "$ROOT/kernel/vinix-mini.elf" "$STAGE/boot/"
# Strip debug info from the kernel copy: keeps the image lean, symbols stay
# in kernel/vinix-mini.elf for crash triage.
strip --strip-debug "$STAGE/boot/vinix-mini.elf" 2>/dev/null || true
cp "$GGUF" "$STAGE/models/model.gguf"
cp "$ROOT/models/tokenizer.json" "$STAGE/models/" 2>/dev/null || echo '{"warn":"tokenizer.json missing, BPE falls back to stub"}'
du -sh "$STAGE" | awk '{print "stage size: " $1}'
# limine.cfg boots straight into inference: no menu timeout, serial console.
cat > "$STAGE/limine.cfg" <<'EOF'
TIMEOUT=0
SERIAL=yes
/VLLM-OS
  PROTOCOL=limine
  KERNEL_PATH=boot:///boot/vinix-mini.elf
  CMDLINE=model=/models/model.gguf port=8000 ctx=4096
EOF
xorriso -as mkisofs -b limine-bios-cd.bin -no-emul-boot -boot-load-size 4 \
  -boot-info-table --efi-boot limine-uefi-cd.bin -efi-boot-part \
  --protective-msdos-label "$STAGE" -o "$OUT"
limine bios-install "$OUT"
ls -lh "$OUT"
