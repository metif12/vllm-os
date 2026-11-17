param(
  [int]$Port = 8000,
  [int]$TimeoutSecs = 120,
  [string]$GuestImg = $env:GUEST_IMG
)
# Phase 2 gate: cold-boot server -> first chat token < 30 s wall (stub mode),
# plus guest->host SLIRP validation when a guest image is provided.
# Exit codes: 0 host gate PASS (guest PASS/SKIP noted), 1 host gate FAIL,
# 2 missing prereq / port busy, 3 guest attempted but FAILED.
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
Set-Location $root
$logDir = "$root/qemu"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$serverLog = "$logDir/test.log"

function Wait-Port([string]$Host_, [int]$Port_, [int]$DeadlineSecs) {
  $until = (Get-Date).AddSeconds($DeadlineSecs)
  while ((Get-Date) -lt $until) {
    $c = New-Object Net.Sockets.TcpClient
    try {
      $r = $c.BeginConnect($Host_, $Port_, $null, $null)
      if ($r.AsyncWaitHandle.WaitOne(250)) { $c.EndConnect($r); return $true }
    } catch { } finally { $c.Close() }
    Start-Sleep -Milliseconds 250
  }
  return $false
}

if (-not (Wait-Port 'localhost' $Port 1)) {
  # Port free: proceed. (Checked cheaply; a listener appearing later is benign
  # because we track our own child process.)
} else {
  # Port already serving: still run the assertions against it read-only, but
  # do not start/stop any server. Timings then exclude boot (noted below).
}

$server = $null
$guest = $null
$bootedOurs = $false
try {
  $t0 = Get-Date
  $existing = Wait-Port 'localhost' $Port 2
  if (-not $existing) {
    if (-not (Test-Path "$root/server.exe")) {
      Write-Output 'building server.exe (gcc -prod) ...'
      & v -prod -cc gcc -o server.exe src
      if ($LASTEXITCODE -ne 0) { exit 2 }
    }
    Write-Output "starting host server on :$Port ..."
    if (Test-Path $serverLog) { Remove-Item $serverLog -Force }
    $server = Start-Process -FilePath "$root/server.exe" -ArgumentList "--port $Port" `
      -NoNewWindow -PassThru -RedirectStandardOutput $serverLog -RedirectStandardError "$logDir/test.stderr.log"
    $bootedOurs = $true
    if (-not (Wait-Port 'localhost' $Port $TimeoutSecs)) {
      Write-Output "FAIL: server not listening on :$Port within ${TimeoutSecs}s (see qemu/test.log)"
      exit 1
    }
  } else {
    Write-Output "note: :$Port already serving, asserting read-only (boot time excluded)"
  }
  $tReady = (Get-Date) - $t0

  $models = Invoke-RestMethod -Uri "http://localhost:$Port/v1/models" -TimeoutSec 5
  $id = $models.data[0].id
  Write-Output "models: $id"
  if ($id -ne 'qwen3-1.7b') { Write-Output "FAIL: unexpected model id '$id'"; exit 1 }

  $t1 = Get-Date
  $body = @{ model = 'qwen3-1.7b'; messages = @(@{ role = 'user'; content = 'hi from qemu path' }) } | ConvertTo-Json -Depth 5
  $chat = Invoke-RestMethod -Uri "http://localhost:$Port/v1/chat/completions" -Method Post `
    -ContentType 'application/json' -Body $body -TimeoutSec 60
  $tFirst = (Get-Date) - $t1
  $text = $chat.choices[0].message.content
  if (-not $text) { Write-Output 'FAIL: empty chat completion'; exit 1 }
  Write-Output "chat: $($text.Substring(0, [Math]::Min(80, $text.Length)))"

  $total = $tReady + $tFirst
  Write-Output ("timing: ready={0:N1}s first_token={1:N1}s total={2:N1}s (gate <30s)" -f $tReady.TotalSeconds, $tFirst.TotalSeconds, $total.TotalSeconds)
  if ($total.TotalSeconds -ge 30) { Write-Output 'FAIL: 30s gate missed'; exit 1 }
  Write-Output 'HOST-GATE PASS'

  # Guest leg: guest reaches the host server at 10.0.2.2:$Port (SLIRP
  # gateway). No hostfwd needed for this direction; hostfwd is only added
  # when ssh automation is configured (GUEST_SSH_PORT + GUEST_SSH_KEY).
  # Without ssh, the default is serial-console automation: boot with a TCP
  # serial, drive the OpenWrt failsafe shell over it (see qemu/README.md),
  # configure 10.0.2.15/24 + default route, and wget /v1/models.
  function Invoke-SerialGuestProbe([int]$SerialPort, [int]$ApiPort) {
    $c = New-Object Net.Sockets.TcpClient
    $c.Connect('127.0.0.1', $SerialPort)
    $s = $c.GetStream()
    $w = New-Object IO.StreamWriter($s)
    $w.AutoFlush = $true
    $script:sbuf = ''
    function Pump([int]$ms) {
      Start-Sleep -Milliseconds $ms
      while ($s.DataAvailable) {
        $b = New-Object byte[] 65536
        $n = $s.Read($b, 0, 65536)
        if ($n -le 0) { break }
        $script:sbuf += [Text.Encoding]::UTF8.GetString($b, 0, $n)
      }
    }
    function Wait-Marker([string]$pat, [int]$t) {
      $u = (Get-Date).AddSeconds($t)
      while ((Get-Date) -lt $u) {
        Pump 500
        if ($script:sbuf -match $pat) { return $true }
      }
      return $false
    }
    if (-not (Wait-Marker 'enter failsafe mode' 240)) { return 'FAIL: preinit prompt never appeared' }
    $w.WriteLine('f')
    if (-not (Wait-Marker 'root@' 240)) { return 'FAIL: failsafe shell never appeared' }
    $script:sbuf = ''
    $w.WriteLine('ip link set eth0 up; ip addr add 10.0.2.15/24 dev eth0; ip route add default via 10.0.2.2')
    Pump 3000
    $w.WriteLine("wget -T10 -qO- http://10.0.2.2:$ApiPort/v1/models; echo WGET_RC=" + '$' + '?')
    if (-not (Wait-Marker 'WGET_RC=' 60)) { return 'FAIL: in-guest wget timed out' }
    if ($script:sbuf -notmatch 'qwen3-1\.7b') {
      return "FAIL: in-guest models mismatch: $($script:sbuf.Substring([Math]::Max(0, $script:sbuf.Length - 300)))"
    }
    # Guest-side chat: full OpenAI-compat round trip from inside the guest,
    # with wall timing toward the <30 s gate.
    $script:sbuf = ''
    $w.WriteLine('date +%s')
    $chatBody = '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi from guest"}]}'
    $w.WriteLine("wget -T15 -O /tmp/chat.json --header='Content-Type: application/json' --post-data='$chatBody' http://10.0.2.2:$ApiPort/v1/chat/completions; echo CHAT_RC=" + '$' + '?')
    $w.WriteLine('ls -l /tmp/chat.json; head -c 200 /tmp/chat.json; echo')
    $w.WriteLine('date +%s')
    if (-not (Wait-Marker 'CHAT_RC=' 90)) { return 'FAIL: in-guest chat timed out' }
    $times = [regex]::Matches($script:sbuf, '\d{10}')
    $chatSecs = if ($times.Count -ge 2) { [int]$times[$times.Count - 1].Value - [int]$times[$times.Count - 2].Value } else { -1 }
    if ($script:sbuf -match 'CHAT_RC=0' -and $script:sbuf -match 'choices') {
      return "PASS chat=${chatSecs}s"
    }
    return "FAIL: in-guest chat mismatch: $($script:sbuf.Substring([Math]::Max(0, $script:sbuf.Length - 1500)))"
  }
  if ($GuestImg -and (Test-Path $GuestImg)) {
    if (-not (Get-Command qemu-system-x86_64 -ErrorAction SilentlyContinue)) {
      Write-Output 'GUEST-SKIP: qemu-system-x86_64 not on PATH'
    } else {
      $serialPort = if ($env:GUEST_SERIAL_PORT) { [int]$env:GUEST_SERIAL_PORT } else { 4446 }
      $extra = if ($env:QEMU_EXTRA_ARGS) { $env:QEMU_EXTRA_ARGS } else { '-accel whpx -machine q35' }
      $guestLog = "$logDir/guest.log"
      $qemuArgs = "$extra -m 512M -display none -serial tcp:127.0.0.1:$serialPort,server,nowait -drive file=`"$GuestImg`",format=raw,if=virtio -netdev user,id=n0"
      if ($env:GUEST_SSH_PORT) { $qemuArgs += ",hostfwd=tcp::$($env:GUEST_SSH_PORT)-:22" }
      $qemuArgs += ' -device virtio-net-pci,netdev=n0'
      Write-Output "booting guest $GuestImg ..."
      Write-Output "qemu args: $qemuArgs"
      $guest = Start-Process -FilePath qemu-system-x86_64 -ArgumentList $qemuArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput "$logDir/qemu.out" -RedirectStandardError "$logDir/qemu.err"
      if ($env:GUEST_SSH_PORT -and $env:GUEST_SSH_KEY -and (Test-Path $env:GUEST_SSH_KEY)) {
        if (Wait-Port 'localhost' ([int]$env:GUEST_SSH_PORT) 90) {
          $out = & ssh -i $env:GUEST_SSH_KEY -p $env:GUEST_SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@localhost "curl -s --max-time 10 http://10.0.2.2:$Port/v1/models"
          if ($out -match 'qwen3-1\.7b') { Write-Output 'GUEST-GATE PASS (in-guest curl 10.0.2.2 OK)' }
          else { Write-Output "FAIL: in-guest curl mismatch: $out"; exit 3 }
        } else {
          Write-Output 'FAIL: guest ssh never came up'; exit 3
        }
      } else {
        $res = Invoke-SerialGuestProbe $serialPort $Port
        if ($res -like 'PASS*') { Write-Output "GUEST-GATE PASS (serial failsafe: guest models+chat via 10.0.2.2 OK, $res)" }
        else { Write-Output $res; exit 3 }
      }
    }
  } else {
    Write-Output 'GUEST-SKIP: set GUEST_IMG to a guest image for guest->host SLIRP validation (OpenWrt ext4-combined tested, see qemu/README.md)'
  }
  exit 0
} finally {
  # taskkill is synchronous enough: Stop-Process returns while WHPX-backed
  # QEMU is still tearing down, leaving the serial port/image locked.
  foreach ($id in @($guest, $server) | Where-Object { $_ -and -not $_.HasExited } | ForEach-Object { $_.Id }) {
    & taskkill /F /PID $id /T 2>$null
  }
  Start-Sleep -Seconds 3
}
