module api

import cfg
import infer

fn new_test_ctx() &HandlerCtx {
	mut m := infer.load_model_stub(infer.Qwen3Config{})
	return &HandlerCtx{
		model:  &m
		config: cfg.defaults()
	}
}

fn test_match_route_hits() {
	r := match_route('GET', '/health') or {
		assert false, 'expected /health route'
		return
	}
	assert r.path == '/health'
	c := match_route('POST', '/v1/chat/completions') or {
		assert false, 'expected chat route'
		return
	}
	assert c.path == '/v1/chat/completions'
	l := match_route('POST', '/v1/completions') or {
		assert false, 'expected completions route'
		return
	}
	assert l.path == '/v1/completions'
	m := match_route('GET', '/v1/models') or {
		assert false, 'expected models route'
		return
	}
	assert m.path == '/v1/models'
	assert routes().len == 4
}

fn test_match_route_misses() {
	if _ := match_route('GET', '/nope') {
		assert false, 'expected 404 miss'
	} else {
		assert true
	}
	if _ := match_route('DELETE', '/health') {
		assert false, 'expected method mismatch miss'
	} else {
		assert true
	}
}

fn test_health_payload() {
	res := handle_health('', new_test_ctx())
	assert res.status == 200
	assert res.body.contains('"status":"ok"')
	assert res.body.contains('Qwen/Qwen3-1.7B')
}

fn test_models_payload_shape() {
	res := handle_models('', new_test_ctx())
	assert res.status == 200
	assert res.body.contains('"object":"list"')
	assert res.body.contains('qwen3-1.7b')
}

fn test_request_sampler_config_fallback() {
	ctx := new_test_ctx()
	base := ctx.config.sampler_params()
	sp := request_sampler(0.7, 40, 0.9, base)
	assert sp.temp == f32(0.7)
	assert sp.top_k == 40
	assert sp.top_p == f32(0.9)
}

fn test_request_sampler_explicit_wins() {
	ctx := new_test_ctx()
	base := ctx.config.sampler_params()
	sp := request_sampler(0.2, 40, 0.9, base)
	assert sp.temp == f32(0.2)
}

fn test_request_max_tokens_clamps() {
	ctx := new_test_ctx()
	assert request_max_tokens(9999, ctx) == 256
	assert request_max_tokens(10, ctx) == 10
	assert request_max_tokens(0, ctx) == 1
}

fn test_bad_chat_body_400() {
	res := handle_chat('not json', new_test_ctx())
	assert res.status == 400
}

fn test_chat_roundtrip_200() {
	body := '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi"}],"stream":false}'
	res := handle_chat(body, new_test_ctx())
	assert res.status == 200
	assert res.body.contains('chat.completion')
}

fn test_completions_roundtrip_200() {
	body := '{"model":"qwen3-1.7b","prompt":"hello"}'
	res := handle_completions(body, new_test_ctx())
	assert res.status == 200
	assert res.body.contains('text_completion')
}

fn test_authorized_open_mode() {
	assert authorized('GET /v1/models HTTP/1.1', '') == true
	assert authorized('', '') == true
}

fn test_authorized_bearer() {
	head := 'GET /v1/models HTTP/1.1\r\nAuthorization: Bearer s3cret\r\n'
	assert authorized(head, 's3cret') == true
	assert authorized(head, 'wrong') == false
	assert authorized('GET /v1/models HTTP/1.1', 's3cret') == false
	assert authorized('GET /x HTTP/1.1\r\nAuthorization: Basic abc', 'abc') == false
	assert authorized('GET /x HTTP/1.1\r\nauthorization: bearer s3cret', 's3cret') == true
}

fn test_bucket_burst_and_refill() {
	mut b := Bucket{
		tokens:  2.0
		updated: 100
	}
	assert b.allow_at(100, 100, 2) == true
	assert b.allow_at(100, 100, 2) == true
	assert b.allow_at(100, 100, 2) == false
	assert b.allow_at(102, 100, 2) == true
	mut c := Bucket{
		tokens:  1.0
		updated: 50
	}
	assert c.allow_at(50, 1, 0) == true
	assert c.allow_at(50, 1, 0) == false
	assert c.allow_at(51, 1, 0) == true
}

fn test_rate_allow_independent_keys() {
	mut buckets := map[string]Bucket{}
	assert allow(mut buckets, 't-key-a', 1000, 1, 1) == true
	assert allow(mut buckets, 't-key-a', 1000, 1, 1) == false
	assert allow(mut buckets, 't-key-b', 1000, 1, 1) == true
	assert allow(mut buckets, 't-key-a', 1001, 1, 1) == true
}

fn test_config_rejects_bad_security_knobs() {
	mut c := cfg.defaults()
	c.server.rate_per_sec = -1
	if _ := c.validate() {
		assert false, 'expected rate_per_sec rejection'
	} else {
		assert true
	}
	c.server.rate_per_sec = 10
	c.server.rate_burst = -2
	if _ := c.validate() {
		assert false, 'expected rate_burst rejection'
	} else {
		assert true
	}
	c.server.rate_burst = 20
	c.validate() or { assert false, 'valid security knobs rejected: ${err}' }
}
