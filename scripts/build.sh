#!/usr/bin/env sh
# Build profiles for vllm-os. Usage: sh scripts/build.sh [host-release]
#
# Every profile uses an optimizing C compiler (-cc gcc); tcc is never
# used. -prod means -O3 + asserts stripped. See scripts/build.ps1 for
# the profile contract (this file mirrors it for linux/mac).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${1:-host-release}"
cd "$ROOT"
case "$PROFILE" in
  host-debug)   v -cc gcc -g -o server src ;;
  host-release) v -prod -cc gcc -o server src ;;
  host-native)  v -prod -cc gcc -cflags "-O3 -fno-strict-aliasing -flto" -o server src ;;
  host-minimal) VLLM_NO_SIMD=1 v -cc gcc -cflags "-Os" -o server src ;;
  qemu)         v -prod -cc gcc -o server src; echo 'run ./scripts/qemu_test.sh next' ;;
  bare-metal)
    echo 'experimental: freestanding kernel-port build'
    v -prod -gc none -freestanding -cc gcc -o kernel/vinix-mini.elf src ;;
  *) echo "unknown profile: $PROFILE (host-debug|host-release|host-native|host-minimal|qemu|bare-metal)" >&2; exit 2 ;;
esac
