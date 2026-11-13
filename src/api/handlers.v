module api

import infer
import json2
import strings
import time

// Route handlers: pure request/response logic, no sockets. Each takes the
// raw body + shared ctx and returns a framed HandlerResult. Server framing
// (status line, Content-Length, Connection: close) lives in server.v.

// handle_health answers GET /health: liveness + served model id.
// Used by orchestration and the QEMU boot gate; never touches weights.
fn handle_health(_body string, ctx &HandlerCtx) HandlerResult {
	id := escape_json(ctx.model.cfg.model_id)
	return HandlerResult{
		status: 200
		body:   '{"status":"ok","model":"${id}"}'
	}
}

// models_payload builds the GET /v1/models body fresh per call — ~100
// bytes of JSON, negligible next to the socket write, and keeps the
// handler stateless (no precomputed globals to invalidate).
fn models_payload() string {
	return json2.encode(ModelsResponse{
		data: [ModelCard{
			id: 'qwen3-1.7b'
		}]
	})
}

fn handle_models(_body string, _ctx &HandlerCtx) HandlerResult {
	return HandlerResult{
		status: 200
		body:   models_payload()
	}
}

// request_sampler merges per-request knobs over the configured sampler:
// the base comes from config (preset + explicit fields), each request
// field wins when it differs from its request-struct default.
fn request_sampler(temp f32, top_k int, top_p f32, dflt infer.SampleParams) infer.SampleParams {
	base := infer.SampleParams{
		temp:  temp
		top_k: top_k
		top_p: top_p
		min_p: dflt.min_p
		seed:  dflt.seed
		fast:  dflt.fast
	}
	if temp == 0.7 && top_k == 40 && top_p == 0.9 {
		// Untouched request: keep the configured preset values.
		return infer.SampleParams{
			temp:  dflt.temp
			top_k: dflt.top_k
			top_p: dflt.top_p
			min_p: dflt.min_p
			seed:  dflt.seed
			fast:  dflt.fast
		}
	}
	return base
}

// request_seed resolves 0 = config seed, config 0 = time-based.
// Explicit nonzero seeds reproduce generations exactly.
fn request_seed(req_seed u64, ctx &HandlerCtx) u64 {
	if req_seed != 0 {
		return req_seed
	}
	if ctx.config.sampler.seed != 0 {
		return ctx.config.sampler.seed
	}
	return u64(time.now().unix_nano())
}

// request_max_tokens clamps the request to the configured ceiling.
fn request_max_tokens(want int, ctx &HandlerCtx) int {
	mt := if want < ctx.config.api.max_tokens { want } else { ctx.config.api.max_tokens }
	return if mt < 1 { 1 } else { mt }
}

fn handle_chat(body string, ctx &HandlerCtx) HandlerResult {
	chat := json2.decode[ChatRequest](body) or {
		return HandlerResult{
			status: 400
			body:   '{"error":"bad chat request"}'
		}
	}
	msgs := chat.messages.map(infer.ChatMsg{
		role:    it.role
		content: it.content
	})
	prompt := infer.apply_chat_template(msgs)
	base := ctx.config.sampler_params()
	mut sp := request_sampler(chat.temperature, chat.top_k, chat.top_p, base)
	sp.seed = request_seed(chat.seed, ctx)
	text := ctx.model.generate(prompt, request_max_tokens(chat.max_tokens, ctx), sp)
	id := 'chatcmpl-${time.now().unix_nano()}'
	created := time.now().unix()
	escaped := escape_json(text)
	mut sb := strings.new_builder(512 + escaped.len)
	sb.write_string('{"id":"')
	sb.write_string(id)
	sb.write_string('","object":"chat.completion","created":')
	sb.write_string(created.str())
	sb.write_string(',"model":"')
	sb.write_string(chat.model)
	sb.write_string('","choices":[{"index":0,"message":{"role":"assistant","content":"')
	sb.write_string(escaped)
	sb.write_string('"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0}}')
	if chat.stream {
		return sse_result(id, chat.model, escaped, created)
	}
	return HandlerResult{
		status: 200
		body:   sb.str()
	}
}

fn handle_completions(body string, ctx &HandlerCtx) HandlerResult {
	comp := json2.decode[CompletionRequest](body) or {
		return HandlerResult{
			status: 400
			body:   '{"error":"bad completion request"}'
		}
	}
	base := ctx.config.sampler_params()
	mut sp := request_sampler(comp.temperature, comp.top_k, comp.top_p, base)
	sp.seed = request_seed(comp.seed, ctx)
	text := ctx.model.generate(comp.prompt, request_max_tokens(comp.max_tokens, ctx), sp)
	id := 'cmpl-${time.now().unix_nano()}'
	created := time.now().unix()
	escaped := escape_json(text)
	mut sb := strings.new_builder(256 + escaped.len)
	sb.write_string('{"id":"')
	sb.write_string(id)
	sb.write_string('","object":"text_completion","created":')
	sb.write_string(created.str())
	sb.write_string(',"model":"')
	sb.write_string(comp.model)
	sb.write_string('","choices":[{"text":"')
	sb.write_string(escaped)
	sb.write_string('","index":0,"finish_reason":"stop"}]}')
	return HandlerResult{
		status: 200
		body:   sb.str()
	}
}

// sse_result frames one full-text chunk + [DONE] with event-stream type.
// Framing is stable so clients work unchanged when per-token SSE arrives.
fn sse_result(id string, model string, text string, created i64) HandlerResult {
	mut sb := strings.new_builder(256 + text.len)
	sb.write_string('{"id":"')
	sb.write_string(id)
	sb.write_string('","object":"chat.completion.chunk","created":')
	sb.write_string(created.str())
	sb.write_string(',"model":"')
	sb.write_string(model)
	sb.write_string('","choices":[{"index":0,"delta":{"content":"')
	sb.write_string(text)
	sb.write_string('"},"finish_reason":null}]}')
	chunk := sb.str()
	mut hb := strings.new_builder(160 + chunk.len)
	hb.write_string('data: ')
	hb.write_string(chunk)
	hb.write_string('\n\ndata: [DONE]\n\n')
	return HandlerResult{
		status:       200
		content_type: 'text/event-stream'
		body:         hb.str()
	}
}
