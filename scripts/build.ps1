#!/usr/bin/env pwsh
# Build profiles for vllm-os. Usage: ./scripts/build.ps1 [-Profile host-release]
#
# Every profile uses an optimizing C compiler (-cc gcc); tcc is never
# used. -prod means -O3 + asserts stripped. The C SIMD layer
# (src/infer/vllm_kernels.c) compiles everywhere via scalar fallbacks;
# AVX2+FMA paths activate only under -march=native / -mavx2, so the
# host-release binary stays portable and host-native is tuned for THIS
# machine (never ship it in the ISO).
param([string]$Profile = 'host-release')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent)
switch ($Profile) {
  'host-debug' {
    # Dev loop: gcc, debug info, no -prod (asserts on, C layer OUT —
    # pure-V kernels, fastest builds, same semantics).
    v -cc gcc -g -o server.exe src
  }
  'host-release' {
    # Portable release: gcc -prod -O3, scalar C kernels, runs anywhere.
    v -prod -cc gcc -o server.exe src
  }
  'host-native' {
    # Tuned release: V's -prod set restated (custom -cflags REPLACE V's
    # defaults, so -O3 -fno-strict-aliasing must be repeated) plus -flto
    # for cross-module inlining. Portable: AVX2 comes from function
    # attributes + runtime dispatch, never -march=native (which
    # pessimize V's own codegen — measured 3-5x slower, see docs/perf.md).
    v -prod -cc gcc -cflags '-O3 -fno-strict-aliasing -flto' -o server.exe src
  }
  'host-minimal' {
    # Smallest auditable binary: no -prod, so no C layer is linked
    # (pure-V kernels only) — plus -Os. Runtime SIMD switches are
    # irrelevant here; there is nothing extra to disable.
    v -cc gcc -cflags '-Os' -o server.exe src
  }
  'qemu'         { v -prod -cc gcc -o server.exe src; Write-Output 'run ./scripts/qemu_test.ps1 next' }
  'bare-metal'   {
    Write-Output 'experimental: freestanding kernel-port build'
    v -prod -gc none -freestanding -cc gcc -o kernel/vinix-mini.elf src
  }
  default { Write-Error "unknown profile: $Profile (host-debug|host-release|host-native|host-minimal|qemu|bare-metal)"; exit 2 }
}
