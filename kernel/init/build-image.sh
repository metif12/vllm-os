#!/bin/sh
# kernel/init/build-image.sh — reproduce the Phase-3 vllm-mini boot ISO.
# Runs on WSL2 Ubuntu (or any Debian-based Linux), no sudo needed.
#
# Network: archive.ubuntu.com must be reachable (xorriso bootstrap);
# github.com releases must be reachable for the pinned Limine tarball.
# Both curl and wget below honor http_proxy/https_proxy — on this LAN
# that is http://192.168.2.5:10808 (export it in WSL before running).
# Downloads are cached under $VLLM_CACHE (default ~/.cache/vllm-os).
#
# Usage:
#   sh kernel/init/build-image.sh [--kernel PATH] [--out ISO] [--work DIR]
#
# What it does:
#   1. builds kernel/init/init.c (static, no libc) into the initramfs tree,
#   2. packs img/ -> initramfs.tar -> initramfs.tar.gz (ustar, `./`-prefixed,
#      sbin/init mode 755 — the kernel's initramfs unpacker requires this),
#   3. stages the Limine BIOS+UEFI layout (pinned limine-binary 12.9.3,
#      downloaded once into $CACHE and reused),
#   4. runs xorriso (system one, else bootstrapped via apt-get download +
#      dpkg -x into the work dir) and the limine BIOS installer.
#
# Boot the result with scripts/mini_boot_gate.ps1 (Windows) or:
#   qemu-system-x86_64 -cdrom vllm-mini.iso -m 2G -display none -serial stdio \
#     -netdev user,id=n0,hostfwd=tcp::18000-:8000 -device e1000,netdev=n0
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL="$ROOT/kernel/mini/bin/vinix"
OUT="$ROOT/vllm-mini.iso"
WORK="$ROOT/kernel/init/.work"
IMAGE_ID="mini-smoke-00000001"
LIMINE_VER="12.9.3"
LIMINE_URL="https://github.com/limine-bootloader/limine/releases/download/v${LIMINE_VER}/limine-binary.tar.gz"
CACHE="${VLLM_CACHE:-$HOME/.cache/vllm-os}"
while [ $# -gt 0 ]; do
  case "$1" in
    --kernel) KERNEL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --image-id) IMAGE_ID="$2"; shift 2;;
    *) echo "usage: $0 [--kernel PATH] [--out ISO] [--work DIR] [--image-id ID]" >&2; exit 2;;
  esac
done
if [ ! -f "$KERNEL" ]; then
  echo "kernel not found: $KERNEL (build kernel/mini first, --kernel to override)" >&2
  exit 1
fi
for t in cc tar gzip dd; do
  command -v "$t" >/dev/null || { echo "need $t on PATH" >&2; exit 1; }
done
mkdir -p "$WORK" "$CACHE"

echo "== init =="
INIT="$WORK/init"
cc -static -nostdlib -ffreestanding -fno-builtin -O2 \
  -o "$INIT" "$ROOT/kernel/init/init.c"
chmod 755 "$INIT"

echo "== initramfs =="
IMG="$WORK/img"
rm -rf "$IMG"
mkdir -p "$IMG/sbin"
cp "$INIT" "$IMG/sbin/init"
chmod 755 "$IMG/sbin/init"
printf '%s\n' "$IMAGE_ID" > "$IMG/.vinix-image-id"
dd if=/dev/urandom of="$IMG/.pad.bin" bs=4096 count=1 status=none
(cd "$IMG" && tar -cf "$WORK/initramfs.tar" .)
gzip -kf "$WORK/initramfs.tar"

echo "== limine =="
LTB="$CACHE/limine-binary-${LIMINE_VER}.tar.gz"
if [ ! -f "$LTB" ]; then
  echo "fetching limine-binary ${LIMINE_VER} ..."
  if command -v curl >/dev/null; then
    curl -L --fail -o "$LTB" "$LIMINE_URL"
  else
    wget -O "$LTB" "$LIMINE_URL"
  fi
fi
rm -rf "$WORK/limine"
mkdir -p "$WORK/limine"
tar -xzf "$LTB" -C "$WORK/limine"
LIMDIR="$(dirname "$(find "$WORK/limine" -name limine-bios-cd.bin | head -1)")"
if [ -z "$LIMDIR" ] || [ ! -f "$LIMDIR/limine-bios.sys" ]; then
  echo "limine tarball layout unexpected (no limine-bios-cd.bin)" >&2
  exit 1
fi
if [ ! -x "$WORK/limine-tool" ]; then
  cc -O2 -o "$WORK/limine-tool" "$LIMDIR/limine.c"
fi

echo "== xorriso =="
if command -v xorriso >/dev/null; then
  XORRISO="xorriso"
else
  XR="$WORK/xorroot"
  if [ ! -x "$XR/usr/bin/xorriso" ]; then
    mkdir -p "$CACHE/debs" "$XR"
    (cd "$CACHE/debs" && apt-get download xorriso libisoburn1t64 libisofs6t64 libburn4t64 libreadline8t64)
    for d in "$CACHE"/debs/*.deb; do dpkg -x "$d" "$XR"; done
  fi
  export LD_LIBRARY_PATH="$XR/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  XORRISO="$XR/usr/bin/xorriso"
fi
"$XORRISO" -version | head -1

echo "== iso =="
ISO_ROOT="$WORK/iso-root"
rm -rf "$ISO_ROOT"
mkdir -p "$ISO_ROOT/boot" "$ISO_ROOT/EFI/BOOT"
cp "$KERNEL" "$ISO_ROOT/boot/vinix"
cp "$WORK/initramfs.tar.gz" "$ISO_ROOT/boot/initramfs.tar.gz"
cp "$LIMDIR/limine-bios-cd.bin" "$LIMDIR/limine-bios.sys" \
   "$LIMDIR/limine-uefi-cd.bin" "$ISO_ROOT/boot/"
cp "$LIMDIR/BOOTX64.EFI" "$ISO_ROOT/EFI/BOOT/"
cat > "$ISO_ROOT/boot/limine.conf" <<'EOF'
timeout: 0
verbose: yes

/Vinix
    protocol: limine
    kernel_path: boot():/boot/vinix
    module_path: boot():/boot/initramfs.tar.gz
    resolution: 1024x768x32
    kaslr: no
EOF
"$XORRISO" -as mkisofs \
  -b boot/limine-bios-cd.bin -no-emul-boot -boot-load-size 4 -boot-info-table \
  --efi-boot boot/limine-uefi-cd.bin -efi-boot-part --efi-boot-image \
  --protective-msdos-label \
  "$ISO_ROOT" -o "$OUT" >/dev/null
"$WORK/limine-tool" bios-install "$OUT" >/dev/null
ls -la "$OUT"
echo "image ready: $OUT"
