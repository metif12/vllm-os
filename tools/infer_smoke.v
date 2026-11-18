module main

// tools/infer_smoke.v — greedy smoke: prefill prompt + N decode steps,
// print token ids + decoded text. Same source runs on the host
// (reference numbers) and as /sbin/vllm inside the guest; greedy math
// (temp 0) is deterministic, so the outputs must match exactly.
//
// Usage:
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o /tmp/smoke tools/infer_smoke.v
//   VLLM_THREADS=1 /tmp/smoke models/Qwen3-0.6B-Q4_0.gguf 'Hello world' 8
//
// NOTE: print/println only, no interpolation — the release-V -prod
// cgen quirk drops the str_intp runtime for interpolation in tiny
// module-main binaries (see forward parity notes).
import infer
import os

// emit prints to stdout and mirrors to the guest serial port. In the
// guest stdout is the framebuffer console (invisible on -display none);
// on the host /dev/com1 does not exist and the mirror is skipped.
fn emit(s string) {
	print(s)
	os.write_file('/dev/com1', s) or {}
}

fn main() {
	weights := if os.args.len > 1 { os.args[1] } else { 'models/Qwen3-0.6B-Q4_0.gguf' }
	prompt := if os.args.len > 2 { os.args[2] } else { 'Hello world' }
	n := if os.args.len > 3 { os.args[3].int() } else { 8 }
	m := infer.load_model(weights, infer.Qwen3Config{}) or { panic(err) }
	// Reuse the boot-cached blob when present (load_model caches it);
	// fall back to a direct load for Models built without the cache.
	blob := if m.blob.len > 0 { m.blob } else { infer.load_model_blob(weights) or { panic(err) } }
	mut f := infer.new_forward_ctx(m.cfg, m.gguf, blob) or { panic(err) }
	ids := m.tok.encode(prompt)
	for id in ids {
		infer.embed_gather(blob, m.gguf, m.cfg.hidden, id, mut f.hidden) or { panic(err) }
		f.decode_step() or { panic(err) }
	}
	mut out := []int{}
	for _ in 0 .. n {
		mut work := f.logits.clone()
		mut idx := []int{len: f.logits.len}
		id := infer.sample_into(mut work, mut idx, infer.SampleParams{
			temp: 0.0
		})
		out << id
		infer.embed_gather(blob, m.gguf, m.cfg.hidden, id, mut f.hidden) or { panic(err) }
		f.decode_step() or { panic(err) }
	}
	emit('tok_ids=' + out.str() + '\n')
	emit('tok_text=' + m.tok.decode(out) + '\n')
	emit('INFER-SMOKE-OK\n')
}
