#!/usr/bin/env python3
"""vllm-os perf harness: one command, fingerprinted, comparable numbers.

Runs the V kernel bench, the NumPy baseline, and (optionally) a pinned
llama-bench matrix, then writes results.json + results.csv next to a
machine fingerprint. docs/perf.md stays the curated log; this script
produces the raw data it quotes.

Quiet-box protocol (enforced unless --force): refuse when the box is
loaded, sleep --cooldown between timed sections, repeat short benches
--reps times and keep every rep (no hidden averaging).

Examples:
  python scripts/bench.py                      # V bench + NumPy, 2 reps
  python scripts/bench.py --force              # this box is always loaded
  python scripts/bench.py --llama-bench /path/to/llama-bench --pp 128 -n 64 -t 8
"""
import argparse
import csv
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fingerprint():
    fp = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    fp["cpu"], fp["ram_gb"], fp["cores"] = cpu_info()
    fp["v_version"] = first_line(["v", "version"])
    fp["gcc_version"] = first_line(["gcc", "--version"])
    fp["git_commit"] = first_line(
        ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"])
    fp["git_clean"] = git_clean()
    return fp


def first_line(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           cwd=ROOT)
        out = (r.stdout or r.stderr or "").strip().splitlines()
        return out[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def git_clean():
    try:
        r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() == ""
    except Exception:
        return False


def cpu_info():
    sysname = platform.system()
    try:
        if sysname == "Linux":
            name = "unknown"
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
            mem = 0.0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem = int(line.split()[1]) / 1024 / 1024
                        break
            return name, round(mem, 1), os.cpu_count()
        if sysname == "Windows":
            name = "unknown"
            try:
                import winreg
                with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") \
                        as k:
                    name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
            except Exception:
                pass
            mem = _windows_ram_gb()
            return name, mem, os.cpu_count()
    except Exception:
        pass
    return platform.processor() or "unknown", 0.0, os.cpu_count()


def _windows_ram_gb():
    try:

        class Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        m = Mem()
        m.dwLength = ctypes.sizeof(Mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 1024**3, 1)
    except Exception:
        pass
    return 0.0


def _windows_cpu_pct():
    """1s-sampled total CPU via GetSystemTimes. Returns 0-100 or -1."""
    try:
        idle = ctypes.c_uint64()
        kernel = ctypes.c_uint64()
        user = ctypes.c_uint64()

        def sample():
            ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel),
                ctypes.byref(user))
            return idle.value, kernel.value + user.value

        i0, t0 = sample()
        time.sleep(1.0)
        i1, t1 = sample()
        dt, di = t1 - t0, i1 - i0
        return 100.0 * (dt - di) / dt if dt else -1
    except Exception:
        return -1


def box_load():
    """Returns (loaded: bool, detail: str). Never raises."""
    try:
        if hasattr(os, "getloadavg") and platform.system() != "Windows":
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return load1 > 0.6 * cores, f"loadavg1={load1:.2f} cores={cores}"
        if platform.system() == "Windows":
            pct = _windows_cpu_pct()
            if pct >= 0:
                return pct > 60, f"cpu={pct:.0f}%"
            return False, "load check unavailable (GetSystemTimes failed)"
    except Exception as e:
        return False, f"load check unavailable ({e})"
    return False, "load check unavailable on this platform"


def run(cmd, cwd=ROOT):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                       cwd=cwd)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


V_LINE = re.compile(r"^(?P<bench>(?:v|q4) .+?): (?P<value>[\d.]+)(?P<unit>us|ms)"
                    r"(?P<rest>.*)$")
LLAMA_ROW = re.compile(r"^\|\s*(?P<model>[^|]+?)\s*\|\s*(?P<size>[^|]+?)\s*\|"
                       r"\s*(?P<params>[^|]+?)\s*\|\s*(?P<backend>[^|]+?)\s*\|"
                       r"\s*(?P<threads>[\d]+)\s*\|\s*(?P<test>pp\d+|tg\d+)"
                       r"\s*\|\s*(?P<tps>[\d.]+)")


def parse_v_bench(text):
    recs = []
    for line in text.splitlines():
        m = V_LINE.match(line.strip())
        if m:
            recs.append({"suite": "v", "bench": m.group("bench").strip(),
                         "value": float(m.group("value")),
                         "unit": m.group("unit"),
                         "detail": m.group("rest").strip()})
    return recs


def parse_numpy(text):
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("py "):
            continue
        m = re.match(r"^py (?P<bench>.+?): (?P<value>[\d.]+)(?P<unit>us|ms)",
                     line)
        if m:
            recs.append({"suite": "numpy", "bench": m.group("bench").strip(),
                         "value": float(m.group("value")),
                         "unit": m.group("unit"), "detail": ""})
        else:
            recs.append({"suite": "numpy", "bench": line, "value": -1,
                         "unit": "", "detail": "unparsed"})
    return recs


def parse_llama(text):
    recs = []
    for line in text.splitlines():
        m = LLAMA_ROW.match(line.strip())
        if m and m.group("test") not in ("test",):
            recs.append({"suite": "llama-bench",
                         "bench": m.group("model").strip() + " "
                         + m.group("test").strip(),
                         "value": float(m.group("tps")), "unit": "tok/s",
                         "detail": ("size=" + m.group("size").strip()
                                    + " params=" + m.group("params").strip()
                                    + " threads=" + m.group("threads"))})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="run even when the box is loaded")
    ap.add_argument("--cooldown", type=int, default=5,
                    help="seconds between timed sections")
    ap.add_argument("--reps", type=int, default=2,
                    help="process repeats of each timed section")
    ap.add_argument("--no-vbench", action="store_true")
    ap.add_argument("--no-numpy", action="store_true")
    ap.add_argument("--llama-bench", default="",
                    help="path to llama-bench binary (skipped if absent)")
    ap.add_argument("--models", nargs="*", default=[
        "models/Qwen3-0.6B-Q4_0.gguf", "models/Llama-3.2-1B-Q4_0.gguf"],
        help="GGUF files for the llama-bench matrix")
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("-t", nargs="*", type=int, default=[8])
    ap.add_argument("--out", default="",
                    help="results dir (default bench/results_<ts>)")
    args = ap.parse_args()

    loaded, detail = box_load()
    print(f"box load: {detail}")
    if loaded and not args.force:
        print("refusing: box is loaded (pass --force to override)")
        return 2

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or os.path.join(ROOT, "bench", f"results_{ts}")
    os.makedirs(outdir, exist_ok=True)
    fp = fingerprint()
    with open(os.path.join(outdir, "fingerprint.json"), "w") as f:
        json.dump(fp, f, indent=2)
    print("fingerprint:", json.dumps(fp))

    records = []
    exe = os.path.join(outdir, "vbench" + (".exe" if os.name == "nt"
                                           else ""))

    def section(name):
        print(f"--- {name} (cooldown {args.cooldown}s) ---")
        time.sleep(args.cooldown)

    if not args.no_vbench:
        section("build v bench")
        code, out = run(["v", "-prod", "-cc", "gcc", "-o", exe,
                         os.path.join("tools", "bench_kernels.v")])
        if code != 0 or not os.path.exists(exe):
            print("V bench build failed:\n" + out[-3000:])
            return 1
        for i in range(args.reps):
            section(f"v bench rep {i + 1}/{args.reps}")
            _, out = run([exe])
            records.extend(parse_v_bench(out))

    if not args.no_numpy:
        if shutil.which(sys.executable) and _has_numpy():
            for i in range(args.reps):
                section(f"numpy baseline rep {i + 1}/{args.reps}")
                _, out = run([sys.executable,
                              os.path.join("scripts", "perf_baseline.py"),
                              "--kernels"])
                records.extend(parse_numpy(out))
        else:
            print("skip numpy baseline: numpy not importable")

    if args.llama_bench:
        if not os.path.exists(args.llama_bench):
            print(f"skip llama-bench: {args.llama_bench} not found")
        else:
            for model in args.models:
                mp = model if os.path.isabs(model) else os.path.join(ROOT,
                                                                     model)
                if not os.path.exists(mp):
                    print(f"skip missing model: {model}")
                    continue
                for t in args.t:
                    section(f"llama-bench {os.path.basename(model)} t={t}")
                    _, out = run([args.llama_bench, "-m", mp, "-p",
                                  str(args.pp), "-n", str(args.n), "-t",
                                  str(t)])
                    for r in parse_llama(out):
                        r["detail"] += f" pp={args.pp} n={args.n}"
                        records.append(r)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"fingerprint": fp, "records": records}, f, indent=2)
    with open(os.path.join(outdir, "results.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "bench", "value", "unit",
                                          "detail"])
        w.writeheader()
        w.writerows(records)
    for r in records:
        print(f"{r['suite']:12} {r['bench']:45} {r['value']:>10} {r['unit']}"
              f"  {r['detail']}")
    print(f"wrote {outdir}")
    return 0


def _has_numpy():
    try:
        __import__("numpy")
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
