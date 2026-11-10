#!/usr/bin/env sh
# Phase 3: pin upstream Vinix and prepare the vinix-mini fork.
# Host-only tooling. Run once: sh kernel/fetch_vinix.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Pinned hobby-OS upstream. Update deliberately, never floating.
VINIX_REPO="${VINIX_REPO:-https://github.com/vlang/vinix}"
VINIX_REF="${VINIX_REF:-master}"
DEST="$ROOT/third_party/vinix"
mkdir -p "$ROOT/third_party"
if [ ! -d "$DEST/.git" ]; then
  git clone "$VINIX_REPO" "$DEST"
fi
git -C "$DEST" fetch origin
git -C "$DEST" checkout "$VINIX_REF"
git -C "$DEST" rev-parse HEAD > "$ROOT/kernel/upstream.pin"
echo "vinix pinned at $(cat "$ROOT/kernel/upstream.pin")"
echo 'Next: create kernel/mini/ with VLLM_MINI flag excluding framebuffer/HID/audio/userland.'
