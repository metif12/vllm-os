module main

// Phase 7 quality gates (host-only, not gated in CI — needs weights on disk).
//
// Runs a fixed "wiki-alt" probe corpus through the real forward pass and
// reports:
//   PPL          - perplexity on the probe (absolute, lower is better)
//   KLD          - KL(ref || ours) over the reference top-k support (0 = same)
//   SAME_TOP_P   - fraction of steps whose top-p token sets match the baseline
//   MAXDIFF      - max |p_ref - p_ours| over the compared support
//   VMHWM_KB     - peak resident set (Linux VmHWM; Windows reports current RSS)
//
// Usage (from repo root):
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o qg tools/quality_gates.v
//   ./qg models/Qwen3-0.6B-Q4_0.gguf --baseline-out base.json
//   VLLM_THREADS=4 ./qg models/Qwen3-0.6B-Q4_0.gguf base.json --max-rss-mb 4096
//
// A baseline is just a JSON file of top-k distributions. It can come from a
// trusted engine (llama.cpp) or from a serial (VLLM_THREADS=1) run of this
// tool; in the latter case KLD/Same-top-p/MAXDIFF measure threading drift.
// Thresholds: --max-kld, --max-topp-miss, --max-diff (defaults 1e-3 / 0 / 1e-3).
// Exit 0 = all gates pass, 1 = metric gate fail, 2 = RSS gate fail.
//
// NOTE: print/println only, no string interpolation — same release-V -prod
// cgen quirk as tools/forward_bench.v in small module-main binaries.
import infer
import json
import os
import runtime

pub struct DistStep {
pub:
	ids   []int
	probs []f64
}

const top_k = 16
const top_p = 0.9

// wiki_alt is a fixed, model-agnostic probe (neutral encyclopedic prose).
// Keeping it in-tree makes PPL comparable across runs and machines.
const wiki_alt = 'The city council met on Tuesday to discuss the annual budget. ' +
	'Members reviewed the proposal for a new library and the repair of several roads. ' +
	'The report noted that public transport usage had risen sharply in the last year, ' +
	'while the cost of maintenance had fallen. A final decision was scheduled for the ' +
	'following month, after the committee had consulted local residents and business owners.'

struct Metrics {
	kld        f64
	same_top_p f64
	max_diff   f64
	steps      int
}

// topk_step computes the full softmax ONCE per step and returns both the
// top-k distribution and the target token's probability (for PPL), so the
// expensive 151936-wide exp sweep is never repeated.
fn topk_step(logits []f32, target int) (DistStep, f64) {
	mut max_logit := f64(logits[0])
	for v in logits {
		if f64(v) > max_logit {
			max_logit = f64(v)
		}
	}
	mut exps := []f64{len: logits.len}
	mut total := f64(0)
	for i, v in logits {
		e := math_exp(f64(v) - max_logit)
		exps[i] = e
		total += e
	}
	p_target := if target >= 0 && target < exps.len { exps[target] / total } else { 1e-9 }
	// Select top-k ids by probability (single pass, k << vocab).
	mut ids := []int{len: top_k}
	mut vals := []f64{len: top_k}
	for i in 0 .. top_k {
		ids[i] = -1
		vals[i] = -1
	}
	for i, e in exps {
		p := e / total
		if p > vals[top_k - 1] {
			vals[top_k - 1] = p
			ids[top_k - 1] = i
			for j := top_k - 1; j > 0 && vals[j] > vals[j - 1]; j-- {
				tv := vals[j]
				vals[j] = vals[j - 1]
				vals[j - 1] = tv
				ti := ids[j]
				ids[j] = ids[j - 1]
				ids[j - 1] = ti
			}
		}
	}
	return DistStep{
		ids:   ids
		probs: vals
	}, p_target
}

// math_exp is a small self-contained exp (no libm in the freestanding C layer;
// this tool is host-only but keeps the same numeric path as the sampler's
// exact softmax). Range-reduced Taylor after argument reduction.
fn math_exp(x f64) f64 {
	if x < -700 {
		return 0
	}
	if x > 700 {
		return 1e300
	}
	// exp(x) = 2^k * exp(r), r in [-ln2/2, ln2/2].
	k := int(x / 0.6931471805599453 + 0.5)
	r := x - f64(k) * 0.6931471805599453
	mut sum := f64(1)
	mut term := f64(1)
	for n in 1 .. 24 {
		term *= r / f64(n)
		sum += term
	}
	mut scale := f64(1)
	if k > 0 {
		for _ in 0 .. k {
			scale *= 2
		}
	} else {
		for _ in 0 .. -k {
			scale *= 0.5
		}
	}
	return sum * scale
}

fn prob_of(step DistStep, id int) f64 {
	for i, tid in step.ids {
		if tid == id {
			return step.probs[i]
		}
	}
	return 1e-9 // floor for ids outside the top-k support
}

fn top_p_ids(step DistStep) map[int]bool {
	mut out := map[int]bool{}
	mut cum := f64(0)
	for i, id in step.ids {
		if id < 0 {
			continue
		}
		out[id] = true
		cum += step.probs[i]
		if cum >= top_p {
			break
		}
	}
	return out
}

fn compare(cur []DistStep, base []DistStep) Metrics {
	n := if cur.len < base.len { cur.len } else { base.len }
	if n == 0 {
		return Metrics{}
	}
	mut kld := f64(0)
	mut matched := 0
	mut max_diff := f64(0)
	for i in 0 .. n {
		b := base[i]
		c := cur[i]
		for id in b.ids {
			if id < 0 {
				continue
			}
			p := prob_of(b, id)
			q := prob_of(c, id)
			kld += p * math_log(p / q)
			mut d := p - q
			if d < 0 {
				d = -d
			}
			if d > max_diff {
				max_diff = d
			}
		}
		bs := top_p_ids(b)
		cs := top_p_ids(c)
		if bs.len == cs.len {
			mut same := true
			for id in bs.keys() {
				if !cs[id] {
					same = false
					break
				}
			}
			if same {
				matched++
			}
		}
	}
	return Metrics{
		kld:        kld / f64(n)
		same_top_p: f64(matched) / f64(n)
		max_diff:   max_diff
		steps:      n
	}
}

fn math_log(x f64) f64 {
	// Natural log via atanh series after mantissa extraction (no libm).
	if x <= 0 {
		return -40.0
	}
	mut y := x
	mut k := 0
	for y >= 2 {
		y *= 0.5
		k++
	}
	for y < 0.5 {
		y *= 2
		k--
	}
	z := (y - 1) / (y + 1)
	mut sum := z
	mut z2 := z * z
	mut term := z
	for n in 1 .. 40 {
		term *= z2
		sum += term / f64(2 * n + 1)
	}
	return 2 * sum + f64(k) * 0.6931471805599453
}

fn rss_kb() i64 {
	$if linux {
		status := os.read_file('/proc/self/status') or { return -1 }
		for line in status.split('\n') {
			if line.starts_with('VmHWM:') {
				parts := line.split_whitespace()
				if parts.len >= 2 {
					return parts[1].i64()
				}
			}
		}
		return -1
	} $else {
		// Windows: current RSS (peak not exposed without a Windows API call).
		return i64(runtime.used_memory() or { 0 }) / 1024
	}
}

fn main() {
	if os.args.len < 2 {
		println('usage: quality_gates <model.gguf> [baseline.json] [--baseline-out f] [--max-rss-mb N] [--max-kld X] [--max-topp-miss X] [--max-diff X]')
		exit(1)
	}
	weights := os.args[1]
	mut baseline_path := ''
	mut baseline_out := ''
	mut max_rss_mb := f64(0)
	mut max_kld := 1e-3
	mut max_topp_miss := f64(0)
	mut max_diff := 1e-3
	mut i := 2
	for i < os.args.len {
		a := os.args[i]
		if a == '--baseline-out' && i + 1 < os.args.len {
			baseline_out = os.args[i + 1]
			i += 2
			continue
		}
		if a == '--max-rss-mb' && i + 1 < os.args.len {
			max_rss_mb = os.args[i + 1].f64()
			i += 2
			continue
		}
		if a == '--max-kld' && i + 1 < os.args.len {
			max_kld = os.args[i + 1].f64()
			i += 2
			continue
		}
		if a == '--max-topp-miss' && i + 1 < os.args.len {
			max_topp_miss = os.args[i + 1].f64()
			i += 2
			continue
		}
		if a == '--max-diff' && i + 1 < os.args.len {
			max_diff = os.args[i + 1].f64()
			i += 2
			continue
		}
		if !a.starts_with('--') && baseline_path == '' {
			baseline_path = a
		}
		i++
	}

	m := infer.load_model(weights, infer.Qwen3Config{}) or { panic(err) }
	blob := if m.blob.len > 0 { m.blob } else { infer.load_model_blob(weights) or { panic(err) } }
	ids := m.tok.encode(wiki_alt)
	if ids.len < 2 {
		panic('probe tokenized to fewer than 2 tokens')
	}
	// Size arenas to the whole probe; no per-step allocation in the loop.
	mut f := infer.new_forward_ctx_cap(m.cfg, m.gguf, blob, ids.len) or { panic(err) }
	f.pool = infer.new_worker_pool(infer.parallel_threads())

	mut steps := []DistStep{cap: ids.len - 1}
	mut nll := f64(0)
	for pos in 0 .. ids.len - 1 {
		infer.embed_gather(blob, m.gguf, m.cfg.hidden, ids[pos], mut f.hidden) or { panic(err) }
		f.decode_step() or { panic(err) }
		step, p_target := topk_step(f.logits, ids[pos + 1])
		steps << step
		if p_target > 0 {
			nll -= math_log(p_target)
		}
	}
	ppl := math_exp(nll / f64(steps.len))
	print('steps=')
	println(steps.len.str())
	print('PPL=')
	println(ppl.str())

	if baseline_out != '' {
		os.write_file(baseline_out, json.encode(steps)) or { panic(err) }
		print('baseline_written=')
		println(baseline_out)
	}

	mut failed := false
	if baseline_path != '' {
		println('[quality] reading baseline')
		raw := os.read_file(baseline_path) or { panic(err) }
		println('[quality] decoding baseline')
		base := json.decode([]DistStep, raw) or { panic(err) }
		println('[quality] comparing')
		met := compare(steps, base)
		println('[quality] compare done')
		print('KLD=')
		println(met.kld.str())
		print('SAME_TOP_P=')
		println(met.same_top_p.str())
		print('MAXDIFF=')
		println(met.max_diff.str())
		if met.kld > max_kld || (f64(1) - met.same_top_p) > max_topp_miss || met.max_diff > max_diff {
			failed = true
		}
	}

	kb := rss_kb()
	print('VMHWM_KB=')
	println(kb.str())
	if max_rss_mb > 0 && kb > 0 && f64(kb) > max_rss_mb * 1024 {
		eprintln('RSS gate failed')
		exit(2)
	}
	if failed {
		eprintln('QUALITY-GATES-FAIL')
		exit(1)
	}
	println('QUALITY-GATES-OK')
}
