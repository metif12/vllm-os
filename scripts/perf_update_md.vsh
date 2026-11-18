#!/usr/bin/env -S v run
// scripts/perf_update_md.vsh — rewrite the CI-tracked perf section of
// docs/perf.md from raw bench results.
//
// V port of scripts/perf_update_md.py (vsh-first tooling rule): reads a
// bench.py results.json ({fingerprint, records}) and replaces ONLY the
// text between the PERF-CI-START / PERF-CI-END markers. The curated
// tables above the markers are never touched.
//
// Usage:
//   v run scripts/perf_update_md.vsh <results.json> [--doc docs/perf.md]
//                                       [--run-url URL]
//
// Exit 0 on success (including no-change), 2 when the markers are missing
// or duplicated. Output is byte-identical to the .py version on the same
// input (verified by fixture test, not by faith).
import json
import math
import os

const start_marker = '<!-- PERF-CI-START -->'
const end_marker = '<!-- PERF-CI-END -->'

struct Fingerprint {
	timestamp   string
	os          string
	os_release  string
	arch        string
	python      string
	cpu         string
	ram_gb      f64
	cores       int
	v_version   string
	gcc_version string
	git_commit  string
	git_clean   bool
}

struct Record {
	suite  string
	bench  string
	value  f64
	unit   string
	detail string
}

struct Results {
	fingerprint Fingerprint
	records     []Record
}

struct Row {
	suite  string
	bench  string
	value  f64
	unit   string
	detail string
	key    string
}

// sort_key orders like Python's (SUITE_ORDER[suite], bench): the order
// digit is zero-padded so one lexical compare does the whole job
// (this V's sort only allows a single a.x < b.x).
fn sort_key(suite string, bench string) string {
	o := suite_order(suite)
	mut pad := o.str()
	for pad.len < 3 {
		pad = '0' + pad
	}
	return pad + '_' + bench
}

// fmt_g renders like Python's f'{v:g}': integral floats print without
// a decimal point, the rest use V's shortest round-trip.
fn fmt_g(v f64) string {
	if v == math.trunc(v) && math.abs(v) < 1e15 {
		return i64(v).str()
	}
	return v.str()
}

// fmt_raw renders like Python's str(number): floats keep their '.0'.
// Only used for the fingerprint header line, never for bench values.
fn fmt_raw_f64(v f64) string {
	if v == math.trunc(v) && math.abs(v) < 1e15 {
		return i64(v).str() + '.0'
	}
	return v.str()
}

fn suite_order(s string) int {
	return match s {
		'forward' { 0 }
		'v' { 1 }
		'numpy' { 2 }
		'llama-bench' { 3 }
		else { 99 }
	}
}

fn aggregate(records []Record) []Row {
	mut idx := map[string]int{}
	mut groups := [][]Record{}
	for r in records {
		key := r.suite + '\x00' + r.bench
		if key in idx {
			groups[idx[key]] << r
		} else {
			idx[key] = groups.len
			groups << [r]
		}
	}
	mut rows := []Row{}
	for g in groups {
		units := g.map(it.unit)
		mut uniq := map[string]bool{}
		for u in units {
			uniq[u] = true
		}
		if uniq.len != 1 {
			eprintln('mixed units, aborting')
			exit(1)
		}
		mut best := g[0]
		for r in g[1..] {
			if r.value < best.value {
				best = r
			}
		}
		mut detail := best.detail
		if g.len > 1 {
			if detail != '' {
				detail += ' [best of ${g.len}]'
			} else {
				detail = '[best of ${g.len}]'
			}
		}
		rows << Row{
			suite:  g[0].suite
			bench:  g[0].bench
			value:  best.value
			unit:   best.unit
			detail: detail
			key:    sort_key(g[0].suite, g[0].bench)
		}
	}
	rows.sort(a.key < b.key)
	return rows
}

fn render_section(fp Fingerprint, rows []Row, run_url string) string {
	mut commit := fp.git_commit
	if commit == '' {
		commit = 'unknown'
	}
	if !fp.git_clean {
		commit += ' (dirty tree)'
	}
	mut run := ' (local run)'
	if run_url != '' {
		run = ' ([workflow run](' + run_url + '))'
	}
	mut out := '_Updated ' + fp.timestamp + ' from `' + commit + '` on ' + fp.os + ' (`' + fp.cpu + '`, ' + fp.cores.str() + ' cores, ' + fmt_raw_f64(fp.ram_gb) + ' GB RAM, V ' + fp.v_version + ', gcc ' + fp.gcc_version + ')' + run + '._\n'
	out += '\n'
	out += '_Runner hardware varies between runs: these rows track regressions '
	out += 'on CI, they do NOT replace the curated tables above and must not '
	out += 'be quoted as machine numbers._\n'
	out += '\n'
	out += '| suite | bench | best | unit | detail |\n'
	out += '|---|---|---|---|---|\n'
	for r in rows {
		out += '| ' + r.suite + ' | ' + r.bench + ' | ' + fmt_g(r.value) + ' | ' + r.unit + ' | ' + r.detail + ' |\n'
	}
	return out
}

fn main() {
	mut results := ''
	mut doc := ''
	mut run_url := ''
	mut i := 1
	for i < os.args.len {
		a := os.args[i]
		if a == '--doc' {
			i++
			doc = os.args[i]
		} else if a == '--run-url' {
			i++
			run_url = os.args[i]
		} else if a == '--help' || a == '-h' {
			println('usage: v run scripts/perf_update_md.vsh <results.json> [--doc docs/perf.md] [--run-url URL]')
			exit(0)
		} else if results == '' {
			results = a
		} else {
			eprintln('unexpected argument: ' + a)
			exit(2)
		}
		i++
	}
	if results == '' {
		eprintln('usage: v run scripts/perf_update_md.vsh <results.json> [--doc docs/perf.md] [--run-url URL]')
		exit(2)
	}
	if doc == '' {
		eprintln('missing --doc (refusing to guess the repo layout)')
		exit(2)
	}
	data := json.decode(Results, os.read_file(results) or {
		eprintln('cannot read results: ' + err.msg())
		exit(2)
	}) or {
		eprintln('bad results.json: ' + err.msg())
		exit(2)
	}
	rows := aggregate(data.records)
	section := render_section(data.fingerprint, rows, run_url)
	text := os.read_file(doc) or {
		eprintln('cannot read doc: ' + err.msg())
		exit(2)
	}
	if text.count(start_marker) != 1 || text.count(end_marker) != 1 {
		eprintln('error: need exactly one ' + start_marker + ' and one ' + end_marker + ' in ' + doc)
		exit(2)
	}
	before := text.split(start_marker)[0]
	after := text.split(end_marker)[1]
	new_doc := before + start_marker + '\n' + section + end_marker + after
	if new_doc == text {
		println('perf.md CI section already up to date')
		exit(0)
	}
	os.write_file(doc, new_doc) or {
		eprintln('cannot write doc: ' + err.msg())
		exit(2)
	}
	println('updated ' + doc + ': ' + rows.len.str() + ' rows')
}
