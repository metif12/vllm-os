param(
  [string]$ISO = "$PSScriptRoot\..\vllm-os\vllm-mini.iso",
  [int]$HostPort = 18000,
  [string]$LogDir = "$env:TEMP\opencode"
)
# Phase-3 boot gate: vllm-mini ISO under QEMU.
# Asserts serial `[vllm-os] boot ok` + TCP SYN answered on guest :8000
# (via SLIRP hostfwd) with a full HTTP round trip. Exit 0 gate green,
# 1 gate red, 2 prereq issue.
$ErrorActionPreference = 'Stop'
$QEMU = 'C:\Program Files\qemu\qemu-system-x86_64.exe'
if (-not (Test-Path $QEMU)) { echo "MISSING: $QEMU"; exit 2 }
if (-not (Test-Path $ISO)) { echo "MISSING ISO: $ISO (sh kernel/init/build-image.sh --out $ISO)"; exit 2 }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LOG  = Join-Path $LogDir 'mini-boot.log'
$QERR = Join-Path $LogDir 'mini-qemu.err'
Remove-Item $LOG, $QERR -ErrorAction SilentlyContinue
$p = Start-Process -NoNewWindow -PassThru -FilePath $QEMU -ArgumentList @(
  '-cdrom', $ISO, '-m', '2G', '-display', 'none',
  '-serial', "file:$LOG",
  '-netdev', "user,id=n0,hostfwd=tcp::${HostPort}-:8000",
  '-device', 'e1000,netdev=n0') -RedirectStandardError $QERR
try {
  # Wait for the guest stack (DHCP lease) before touching the port: a
  # host-side TCP connect can complete at QEMU's hostfwd layer while the
  # guest still has no IP, which looks like a hung server.
  $leased = $false
  for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { echo "QEMU-EXITED-EARLY code=$($p.ExitCode)"; break }
    if ((Test-Path $LOG) -and (Select-String -Path $LOG -Pattern 'DHCP lease' -Quiet)) { $leased = $true; break }
  }
  echo "DHCP-LEASED=$leased"
  $ok = $false
  $body = $null
  if ($leased) {
  for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { echo "QEMU-EXITED-EARLY code=$($p.ExitCode)"; break }
    try {
      $c = New-Object Net.Sockets.TcpClient
      $iar = $c.BeginConnect('127.0.0.1', $HostPort, $null, $null)
      if ($iar.AsyncWaitHandle.WaitOne(1500) -and $c.Connected) {
        $ok = $true
        $s = $c.GetStream()
        $req = [Text.Encoding]::ASCII.GetBytes("GET / HTTP/1.0`r`n`r`n")
        $s.Write($req, 0, $req.Length)
        $buf = New-Object byte[] 4096
        $s.ReadTimeout = 5000
        $sb = New-Object Text.StringBuilder
        try {
          while (($n = $s.Read($buf, 0, $buf.Length)) -gt 0) {
            $sb.Append([Text.Encoding]::ASCII.GetString($buf, 0, $n)) | Out-Null
          }
        } catch {}
        $body = $sb.ToString()
        $c.Close()
        break
      }
      $c.Close()
    } catch {}
  }
  }
  echo "SYN-ANSWERED=$ok"
  if ($body) { echo "--- HTTP reply ---"; echo $body }
} finally {
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
}
Start-Sleep -Seconds 1
$bootOk = $false; $listen = $false; $noPanic = $false
if (Test-Path $LOG) {
  echo '--- serial tail ---'
  Get-Content $LOG -Tail 12
  $t = Get-Content $LOG -Raw
  $bootOk = $t -match '\[vllm-os\] boot ok'
  $listen = $t -match 'listening on :8000'
  $noPanic = $t -notmatch 'Could not start init process'
  echo '--- asserts ---'
  echo "BOOT-OK=$bootOk"
  echo "LISTEN=$listen"
  echo "NO-INIT-PANIC=$noPanic"
} else { echo 'NO-SERIAL-LOG' }
if (Test-Path $QERR) { $e = Get-Content $QERR -Raw; if ($e) { echo '--- qemu stderr ---'; echo $e } }
$httpOk = ($body -match '"boot":"ok"')
if ($leased -and $ok -and $httpOk -and $bootOk -and $listen -and $noPanic) {
  echo 'MINI-BOOT-GATE PASS'
  exit 0
} else {
  echo 'MINI-BOOT-GATE FAIL'
  exit 1
}
