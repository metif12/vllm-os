module infer

fn test_kvcache_append_row_reset() {
	mut c := new_kv_cache(2, 4)!
	assert c.len() == 0
	c.append([f32(1), 2])!
	c.append([f32(3), 4])!
	assert c.len() == 2
	assert c.row(0)! == [f32(1), 2]
	assert c.row(1)! == [f32(3), 4]
	if _ := c.row(2) {
		assert false, 'expected row bounds error'
	} else {
		assert true
	}
	if _ := c.append([f32(9)]) {
		assert false, 'expected dim mismatch error'
	} else {
		assert true
	}
	c.reset()
	assert c.len() == 0
	c.append([f32(5), 6])!
	c.append([f32(7), 8])!
	c.append([f32(9), 10])!
	c.append([f32(11), 12])!
	if _ := c.append([f32(13), 14]) {
		assert false, 'expected full-cache error'
	} else {
		assert true
	}
}

fn test_new_kv_cache_rejects_bad_shape() {
	if _ := new_kv_cache(0, 4) {
		assert false, 'expected dim rejection'
	} else {
		assert true
	}
	if _ := new_kv_cache(2, 0) {
		assert false, 'expected cap rejection'
	} else {
		assert true
	}
}

fn test_attention_step_hand_values() {
	mut k := new_kv_cache(2, 4)!
	mut v := new_kv_cache(2, 4)!
	k.append([f32(1), 0])!
	k.append([f32(0), 1])!
	v.append([f32(10), 20])!
	v.append([f32(30), 40])!
	q := [f32(1), 0]
	mut probs := []f32{len: 2}
	mut out := []f32{len: 2}
	attention_step(q, &k, &v, 1.0, mut probs, mut out)!
	// scores [1,0] -> softmax [0.7311, 0.2689];
	// out = [0.7311*10 + 0.2689*30, 0.7311*20 + 0.2689*40].
	mut ps := f32(0)
	for p in probs {
		ps += p
	}
	assert ps > 0.999 && ps < 1.001
	assert out[0] > 15.37 && out[0] < 15.39
	assert out[1] > 25.37 && out[1] < 25.39
}

fn test_attention_step_errors() {
	mut k := new_kv_cache(2, 4)!
	mut v := new_kv_cache(2, 4)!
	k.append([f32(1), 0])!
	mut probs := []f32{len: 2}
	mut out := []f32{len: 2}
	// kv length mismatch (v empty)
	if _ := attention_step([f32(1), 0], &k, &v, 1.0, mut probs, mut out) {
		assert false, 'expected kv mismatch error'
	} else {
		assert true
	}
	v.append([f32(1), 1])!
	v.append([f32(2), 2])!
	k.append([f32(0), 1])!
	// dim mismatch
	if _ := attention_step([f32(1), 0, 0], &k, &v, 1.0, mut probs, mut out) {
		assert false, 'expected dim error'
	} else {
		assert true
	}
	// short buffers (n=2 rows, 1-elem scratch)
	mut short_p := []f32{len: 1}
	if _ := attention_step([f32(1), 0], &k, &v, 1.0, mut short_p, mut out) {
		assert false, 'expected probs error'
	} else {
		assert true
	}
	mut short_o := []f32{len: 1}
	if _ := attention_step([f32(1), 0], &k, &v, 1.0, mut probs, mut short_o) {
		assert false, 'expected out error'
	} else {
		assert true
	}
	// empty cache
	mut ke := new_kv_cache(2, 4)!
	mut ve := new_kv_cache(2, 4)!
	if _ := attention_step([f32(1), 0], &ke, &ve, 1.0, mut probs, mut out) {
		assert false, 'expected empty-cache error'
	} else {
		assert true
	}
}
