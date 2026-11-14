#!/usr/bin/env pwsh
# Build profiles for vllm-os. Usage: ./scripts/build.ps1 [-Profile host-release]
#
# Every profile uses an optimizing C compiler (-cc gcc); tcc is never
# used. -prod means -O3 + asserts stripped. The C SIMD layer
# (src/infer/vllm_kernels.c) uses function-target AVX2+FMA + runtime
# dispatch, so the host-release binary is portable AND fastest with no
# custom flags (a separate "native" profile measured identical and was
# retired).
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
    # Portable AND fastest: -prod already passes -O3 -flto; AVX2 comes
    # from function-target attributes + runtime dispatch, so no custom
    # flags are needed (a separate "native" profile measured identical
    # and was retired).
    v -prod -cc gcc -o server.exe src
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
  default { Write-Error "unknown profile: $Profile (host-debug|host-release|host-minimal|qemu|bare-metal)"; exit 2 }
}
