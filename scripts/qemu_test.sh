#!/bin/sh
# Phase 2 gate (sh mirror of qemu_test.ps1): cold-boot server -> first chat
# token < 30 s wall (stub mode), plus guest->host SLIRP validation when
# GUEST_IMG points at a qcow2 image.
# Exit codes: 0 host PASS (guest PASS/SKIP noted), 1 host FAIL,
# 2 missing prereq / port busy, 3 guest attempted but FAILED.
set -eu
PORT="${1:-8000}"
TIMEOUT_SECS="${TIMEOUT_SECS:-120}"
GUEST_IMG="${GUEST_IMG:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/qemu/test.log"
mkdir -p "$ROOT/qemu"

wait_port() { # host port tries(0.25s each)
  i=0
  while [ "$i" -lt "$2" ]; do
    if (echo >/dev/tcp/"$1"/"$PORT") 2>/dev/null; then return 0; fi
    sleep 0.25
    i=$((i + 1))
  done
  return 1
}

SERVER_PID=""
GUEST_PID=""
cleanup() {
  [ -n "$GUEST_PID" ] && kill "$GUEST_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true # reap so ports/image locks release before exit
}
trap cleanup EXIT INT TERM

T0=$(date +%s)
if wait_port localhost 8; then
  echo "note: :$PORT already serving, asserting read-only (boot time excluded)"
  T_READY=0
else
  if [ ! -x "$ROOT/server" ]; then
    echo "building server (gcc -prod) ..."
    v -prod -cc gcc -o server src || exit 2
  fi
  echo "starting host server on :$PORT ..."
  rm -f "$LOG"
  "$ROOT/server" --port "$PORT" >"$LOG" 2>&1 &
  SERVER_PID="$!"
  TRIES=$((TIMEOUT_SECS * 4))
  if ! wait_port localhost "$TRIES"; then
    echo "FAIL: server not listening on :$PORT within ${TIMEOUT_SECS}s (see qemu/test.log)"
    exit 1
  fi
  T_READY=$(( $(date +%s) - T0 ))
fi

ID=$(curl -s --max-time 5 "http://localhost:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1)
echo "models: $ID"
case "$ID" in
  *qwen3-1.7b*) ;;
  *) echo "FAIL: unexpected model id '$ID'"; exit 1 ;;
esac

T1=$(date +%s)
CHAT=$(curl -s --max-time 60 -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi from qemu path"}]}')
T_FIRST=$(( $(date +%s) - T1 ))
echo "chat: $(printf '%s' "$CHAT" | head -c 80)"
[ -n "$CHAT" ] || { echo "FAIL: empty chat completion"; exit 1; }

TOTAL=$((T_READY + T_FIRST))
echo "timing: ready=${T_READY}s first_token=${T_FIRST}s total=${TOTAL}s (gate <30s)"
[ "$TOTAL" -lt 30 ] || { echo "FAIL: 30s gate missed"; exit 1; }
echo "HOST-GATE PASS"

# Guest leg: guest reaches host at 10.0.2.2:$PORT (SLIRP gateway, no hostfwd
# needed). hostfwd is added only for ssh automation (GUEST_SSH_PORT/KEY).
# Without ssh, the default is serial-console automation (bash only): boot
# with a TCP serial, drive the OpenWrt failsafe shell over it, configure
# 10.0.2.15/24 + default route, and wget /v1/models.
serial_guest_probe() { # serial_port api_port -> prints result, 0 on PASS
  _sp="$1"; _ap="$2"
  exec 8<>/dev/tcp/127.0.0.1/"$_sp" || return 1
  _buf=""; _t=0
  while [ "$_t" -lt 480 ]; do
    if IFS= read -r -t 1 -u 8 _line; then _buf="$_buf
$_line"; fi
    case "$_buf" in *'enter failsafe mode'*) break ;; esac
    _t=$((_t + 1))
  done
  case "$_buf" in *'enter failsafe mode'*) ;; *) exec 8>&-; echo "FAIL: preinit prompt never appeared"; return 1 ;; esac
  printf 'f\n' >&8
  _buf=""; _t=0
  while [ "$_t" -lt 480 ]; do
    if IFS= read -r -t 1 -u 8 _line; then _buf="$_buf
$_line"; fi
    case "$_buf" in *'root@'*) break ;; esac
    _t=$((_t + 1))
  done
  case "$_buf" in *'root@'*) ;; *) exec 8>&-; echo "FAIL: failsafe shell never appeared"; return 1 ;; esac
  printf 'ip link set eth0 up; ip addr add 10.0.2.15/24 dev eth0; ip route add default via 10.0.2.2\n' >&8
  sleep 3
  printf 'wget -T10 -qO- http://10.0.2.2:%s/v1/models; echo WGET_RC=$?\n' "$_ap" >&8
  _buf=""; _t=0
  while [ "$_t" -lt 60 ]; do
    if IFS= read -r -t 1 -u 8 _line; then _buf="$_buf
$_line"; fi
    case "$_buf" in *'WGET_RC='*) break ;; esac
    _t=$((_t + 1))
  done
  case "$_buf" in *qwen3-1.7b*) ;; *) exec 8>&-; echo "FAIL: in-guest models mismatch"; return 1 ;; esac
  # Guest-side chat: full OpenAI-compat round trip from inside the guest.
  printf 'date +%%s\n' >&8
  printf 'wget -T15 -O /tmp/chat.json --header=%s --post-data=%s http://10.0.2.2:%s/v1/chat/completions; echo CHAT_RC=$?\n' \
    "'Content-Type: application/json" \
    "'{\"model\":\"qwen3-1.7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi from guest\"}]}'" \
    "$_ap" >&8
  printf 'ls -l /tmp/chat.json; head -c 200 /tmp/chat.json; echo\n' >&8
  printf 'date +%%s\n' >&8
  _buf=""; _t=0
  while [ "$_t" -lt 90 ]; do
    if IFS= read -r -t 1 -u 8 _line; then _buf="$_buf
$_line"; fi
    case "$_buf" in *'CHAT_RC='*) break ;; esac
    _t=$((_t + 1))
  done
  exec 8>&-
  case "$_buf" in
    *'CHAT_RC=0'*choices*) echo "PASS chat round trip ok" ;;
    *) echo "FAIL: in-guest chat mismatch: $(printf '%s' "$_buf" | tail -c 300)" ;;
  esac
  return 0
}
if [ -n "$GUEST_IMG" ] && [ -f "$GUEST_IMG" ]; then
  if ! command -v qemu-system-x86_64 >/dev/null; then
    echo "GUEST-SKIP: qemu-system-x86_64 not on PATH"
  else
    SERIAL_PORT="${GUEST_SERIAL_PORT:-4446}"
    EXTRA="${QEMU_EXTRA_ARGS:--accel kvm -machine q35}"
    ARGS="$EXTRA -m 512M -display none -serial tcp:127.0.0.1:$SERIAL_PORT,server,nowait -drive file=$GUEST_IMG,format=raw,if=virtio -netdev user,id=n0"
    if [ -n "${GUEST_SSH_PORT:-}" ]; then ARGS="$ARGS,hostfwd=tcp::$GUEST_SSH_PORT-:22"; fi
    ARGS="$ARGS -device virtio-net-pci,netdev=n0"
    echo "booting guest $GUEST_IMG ..."
    echo "qemu args: $ARGS"
    # shellcheck disable=SC2086
    qemu-system-x86_64 $ARGS >"$ROOT/qemu/qemu.out" 2>"$ROOT/qemu/qemu.err" &
    GUEST_PID="$!"
    if [ -n "${GUEST_SSH_PORT:-}" ] && [ -n "${GUEST_SSH_KEY:-}" ] && [ -f "$GUEST_SSH_KEY" ]; then
      i=0
      while [ "$i" -lt 360 ] && ! (echo >/dev/tcp/localhost/"$GUEST_SSH_PORT") 2>/dev/null; do sleep 0.25; i=$((i+1)); done
      [ "$i" -lt 360 ] || { echo "FAIL: guest ssh never came up"; exit 3; }
      OUT=$(ssh -i "$GUEST_SSH_KEY" -p "$GUEST_SSH_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@localhost "curl -s --max-time 10 http://10.0.2.2:$PORT/v1/models")
      case "$OUT" in
        *qwen3-1.7b*) echo "GUEST-GATE PASS (in-guest curl 10.0.2.2 OK)" ;;
        *) echo "FAIL: in-guest curl mismatch: $OUT"; exit 3 ;;
      esac
    elif [ -n "${BASH_VERSION:-}" ]; then
      RES=$(serial_guest_probe "$SERIAL_PORT" "$PORT") || { echo "$RES"; exit 3; }
      case "$RES" in
        PASS*) echo "GUEST-GATE PASS (serial failsafe: guest models+chat via 10.0.2.2 OK, $RES)" ;;
        *) echo "$RES"; exit 3 ;;
      esac
    else
      echo "GUEST-SKIP: serial automation needs bash (set GUEST_SSH_PORT + GUEST_SSH_KEY for ssh path instead)"
    fi
  fi
else
  echo "GUEST-SKIP: set GUEST_IMG to a guest image for guest->host SLIRP validation (OpenWrt ext4-combined tested, see qemu/README.md)"
fi
