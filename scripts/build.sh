#!/usr/bin/env sh
# Build profiles for vllm-os. Usage: sh scripts/build.sh [host-release]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${1:-host-release}"
cd "$ROOT"
case "$PROFILE" in
  host-debug)   v -o server src ;;
  host-release) v -prod -o server src ;;
  qemu)         v -prod -o server src; echo 'run ./scripts/qemu_test.sh next' ;;
  bare-metal)
    echo 'experimental: freestanding kernel-port build'
    v -gc none -freestanding -o kernel/vinix-mini.elf src ;;
  *) echo "unknown profile: $PROFILE (host-debug|host-release|qemu|bare-metal)" >&2; exit 2 ;;
esac
