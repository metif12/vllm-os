#!/usr/bin/env pwsh
# Build profiles for vllm-os. Usage: ./scripts/build.ps1 [-Profile host-release]
param([string]$Profile = 'host-release')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent)
switch ($Profile) {
  'host-debug'   { v -o server.exe src }
  'host-release' { v -prod -o server.exe src }
  'qemu'         { v -prod -o server.exe src; Write-Output 'run ./scripts/qemu_test.ps1 next' }
  'bare-metal'   {
    Write-Output 'experimental: freestanding kernel-port build'
    v -gc none -freestanding -o kernel/vinix-mini.elf src
  }
  default { Write-Error "unknown profile: $Profile (host-debug|host-release|qemu|bare-metal)"; exit 2 }
}
