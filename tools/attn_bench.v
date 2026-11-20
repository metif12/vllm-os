module main

// Phase 10 attention-scaling probe (host-only, no weights): times one head
// step over full vs windowed attention at 512 and 4096 KV positions.
// The exponential cost of full attention is what the 4k cap protects;
// sliding windows bound it. Reports ms/head-step for both.
//
// Usage:
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o attnbench tools/attn_bench.v
//   ./attnbench
import infer
import time

fn bench_one(n int, dim int, window int, sinks int, reps int) (f64, f64) {
	mut k := infer.new_kv_cache(dim, n) or { panic(err) }
	mut v := infer.new_kv_cache(dim, n) or { panic(err) }
	for i in 0 .. n {
		mut kr := []f32{len: dim}
		mut vr := []f32{len: dim}
		for d in 0 .. dim {
			kr[d] = f32(((i * 31 + d * 7) % 23) - 11) * 0.02
			vr[d] = f32(((i * 17 + d * 3) % 19) - 9) * 0.02
		}
		k.append(kr) or { panic(err) }
		v.append(vr) or { panic(err) }
	}
	mut q := []f32{len: dim}
	for d in 0 .. dim {
		q[d] = f32((d % 7) - 3) * 0.03
	}
	mut out := []f32{len: dim}
	mut probs := []f32{len: n}
	// warm up
	infer.attention_step(q, &k, &v, 0.09, mut probs, mut out) or { panic(err) }
	mut sw := time.new_stopwatch()
	for _ in 0 .. reps {
		infer.attention_step(q, &k, &v, 0.09, mut probs, mut out) or { panic(err) }
	}
	full_us := f64(sw.elapsed().microseconds()) / f64(reps)
	if window > 0 {
		infer.attention_step_window(q, &k, &v, 0.09, window, sinks, mut probs, mut out) or {
			panic(err)
		}
		sw = time.new_stopwatch()
		for _ in 0 .. reps {
			infer.attention_step_window(q, &k, &v, 0.09, window, sinks, mut probs, mut out) or {
				panic(err)
			}
		}
	}
	win_us := f64(sw.elapsed().microseconds()) / f64(reps)
	return full_us, win_us
}

fn main() {
	dim := 128
	reps := 200
	for n in [512, 4096] {
		full_us, win_us := bench_one(n, dim, 256, 4, reps)
		print('n=')
		print(n.str())
		print(' full_us=')
		println(full_us.str())
		print('n=')
		print(n.str())
		print(' window_us=')
		println(win_us.str())
	}
	println('ATTN-BENCH-OK')
}
