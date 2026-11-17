#!/usr/bin/env python3
"""Rewrite the CI-tracked perf section of docs/perf.md from raw results.

Reads a bench.py results.json ({fingerprint, records}) and replaces ONLY
the text between the PERF-CI-START / PERF-CI-END markers. The curated
tables above the markers are never touched: CI runners vary, so CI rows
are tracked separately and must not be quoted as machine numbers.

Usage:
  python scripts/perf_update_md.py <results.json> [--doc docs/perf.md]
                                     [--run-url URL]

Exit 0 on success (including no-change), 2 when the markers are missing
or duplicated.
"""
import argparse
import json
import os
import sys

START = "<!-- PERF-CI-START -->"
END = "<!-- PERF-CI-END -->"
SUITE_ORDER = {"forward": 0, "v": 1, "numpy": 2, "llama-bench": 3}


def fmt_value(v):
    return f"{v:g}"


def aggregate(records):
    """Group by (suite, bench): best (min) value wins, units must agree."""
    groups = {}
    for r in records:
        key = (r["suite"], r["bench"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (suite, bench), rs in sorted(
            groups.items(),
            key=lambda kv: (SUITE_ORDER.get(kv[0][0], 99), kv[0][1])):
        units = {r["unit"] for r in rs}
        if len(units) != 1:
            raise ValueError(f"mixed units for {suite}/{bench}: {units}")
        best = min(rs, key=lambda r: r["value"])
        detail = best.get("detail", "")
        if len(rs) > 1 and detail:
            detail = f"{detail} [best of {len(rs)}]"
        elif len(rs) > 1:
            detail = f"[best of {len(rs)}]"
        rows.append({"suite": suite, "bench": bench,
                     "value": best["value"], "unit": next(iter(units)),
                     "detail": detail})
    return rows


def render_section(fp, rows, run_url):
    commit = fp.get("git_commit", "unknown")
    if not fp.get("git_clean", True):
        commit += " (dirty tree)"
    run = f" ([workflow run]({run_url}))" if run_url else " (local run)"
    lines = [
        "_Updated "
        + f"{fp.get('timestamp', 'unknown')} from `{commit}` on "
        + f"{fp.get('os', '?')} "
        + f"(`{fp.get('cpu', 'unknown cpu')}`, "
        + f"{fp.get('cores', '?')} cores, {fp.get('ram_gb', '?')} GB RAM, "
        + f"V {fp.get('v_version', '?')}, gcc {fp.get('gcc_version', '?')})"
        + run + "._",
        "",
        "_Runner hardware varies between runs: these rows track regressions "
        "on CI, they do NOT replace the curated tables above and must not "
        "be quoted as machine numbers._",
        "",
        "| suite | bench | best | unit | detail |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['suite']} | {r['bench']} | "
                     f"{fmt_value(r['value'])} | {r['unit']} | "
                     f"{r['detail']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="bench.py results.json")
    ap.add_argument("--doc", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "perf.md"), help="perf.md to update")
    ap.add_argument("--run-url", default="",
                    help="workflow run URL for provenance")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    rows = aggregate(data["records"])
    section = render_section(data["fingerprint"], rows, args.run_url)

    with open(args.doc) as f:
        doc = f.read()
    if doc.count(START) != 1 or doc.count(END) != 1:
        print(f"error: need exactly one {START} and one {END} in {args.doc}",
              file=sys.stderr)
        return 2
    before, rest = doc.split(START)
    _, after = rest.split(END)
    new_doc = before + START + "\n" + section + END + after
    if new_doc == doc:
        print("perf.md CI section already up to date")
        return 0
    with open(args.doc, "w") as f:
        f.write(new_doc)
    print(f"updated {args.doc}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
