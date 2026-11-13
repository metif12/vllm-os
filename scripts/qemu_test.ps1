param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
Set-Location $root
if (-not (Get-Command qemu-system-x86_64 -ErrorAction SilentlyContinue)) {
  Write-Output 'qemu-system-x86_64 not on PATH. Install: winget install -e --id SoftwareFreedomConservancy.QEMU'
  exit 2
}
if (-not (Test-Path "$root/server.exe")) {
  Write-Output 'building server.exe (gcc -prod) ...'
  & v -prod -cc gcc -o server.exe src
  if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Output "starting host server on :$Port ..."
$log = "$root/qemu/test.log"
$proc = Start-Process -FilePath "$root/server.exe" -ArgumentList "--port $Port" -NoNewWindow -PassThru -RedirectStandardOutput $log
Start-Sleep -Seconds 2
try {
  # Phase 2 SLIRP path: if $env:GUEST_IMG points at a guest disk image, boot it
  # with hostfwd so the guest reaches the host server at 10.0.2.2:$Port.
  if ($env:GUEST_IMG -and (Test-Path $env:GUEST_IMG)) {
    Write-Output "booting guest $env:GUEST_IMG with hostfwd tcp::$Port-:8000 ..."
    $guest = Start-Process -FilePath qemu-system-x86_64 -ArgumentList "-m 1G -display none -serial stdio -drive file=`"$env:GUEST_IMG`",format=qcow2 -netdev user,id=n0,hostfwd=tcp::$($Port)-:8000 -device virtio-net-pci,netdev=n0" -NoNewWindow -PassThru
    Write-Output 'guest booted (assert 10.0.2.2:8000 from inside the guest; see qemu/README.md)'
  }
  $models = Invoke-RestMethod -Uri "http://localhost:$Port/v1/models" -TimeoutSec 5
  Write-Output "models: $($models.data[0].id)"
  $body = @{ model='qwen3-1.7b'; messages=@(@{ role='user'; content='hi from qemu path' }) } | ConvertTo-Json -Depth 5
  $chat = Invoke-RestMethod -Uri "http://localhost:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 15
  Write-Output "chat: $($chat.choices[0].message.content)"
  Write-Output 'QEMU-NET-PATH OK (host loopback; guest SLIRP 10.0.2.2 validated in phase 2 with guest image)'
} finally {
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
