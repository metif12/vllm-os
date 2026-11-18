#!/bin/sh
# Phase 5: `make image` compatibility wrapper. Run from WSL:
#   sh scripts/build_image.sh [path/to/model.gguf]
# Full knob list lives in GNUmakefile (MODEL, TOKENIZER, CONFIG, V,
# KERNEL, OUT, WORK, EXTRA).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/Qwen3-0.6B-Q4_0.gguf}"
exec make -C "$ROOT" -f GNUmakefile image MODEL="$MODEL"
