module api

import json2
import net
import strings
import time
import infer

// Raw-socket HTTP server on purpose: `veb` pulls in OS deps we cannot
// satisfy on vinix-mini. This stays portable to bare metal.
pub struct Server {
pub:
	port int = 8000
mut:
	model &infer.Model
}

const max_body = 1 << 20 // 1 MB cap: rejects absurd Content-Length before alloc
const max_conns = 64 // backpressure: accept blocks past this, no unbounded spawn

// serve starts the OpenAI-compatible HTTP API on the given port.
pub fn serve(port int, model &infer.Model) ! {
	models_json := json2.encode(ModelsResponse{
		data: [ModelCard{
			id: 'qwen3-1.7b'
		}]
	})
	mut ln := net.listen_tcp(.ip6, ':${port}')!
	eprintln('[vllm-os] openai-compat api on :${port} model=${model.cfg.model_id} quant=${model.cfg.quant}')
	mut sema := chan int{cap: max_conns}
	for {
		mut conn := ln.accept()!
		sema <- 1
		spawn handle(mut conn, model, models_json, sema)
	}
}

fn sema_release(s chan int) {
	token := <-s
	_ = token
}

// read_request reads headers + exactly Content-Length body bytes.
// Single small stack buffer grows only to the declared body size.
fn read_request(mut conn net.TcpConn) !(string, string) {
	mut acc := []u8{cap: 8192}
	mut tmp := []u8{len: 4096}
	mut head_end := -1
	mut total_needed := -1
	for {
		n := conn.read(mut tmp) or { return error('closed') }
		if n <= 0 {
			return error('closed')
		}
		acc << tmp[..n]
		if acc.len > 16384 + max_body {
			return error('request too large')
		}
		if head_end < 0 {
			s := acc.bytestr()
			if idx := s.index('\r\n\r\n') {
				head_end = idx
				total_needed = head_end + 4 + content_length(s[..head_end])
				if total_needed - (head_end + 4) > max_body {
					return error('body too large')
				}
			}
		}
		if head_end >= 0 && acc.len >= total_needed {
			break
		}
	}
	full := acc.bytestr()
	return full[..head_end], full[head_end + 4..total_needed]
}

fn content_length(head string) int {
	for line in head.split_into_lines() {
		if line.len > 15 && (line[..15].to_lower() == 'content-length:') {
			return line[15..].trim_space().int()
		}
	}
	return 0
}

fn handle(mut conn net.TcpConn, model &infer.Model, models_json string, sema chan int) {
	defer {
		sema_release(sema)
		conn.close() or {}
	}
	head, body := read_request(mut conn) or { return }
	lines := head.split_into_lines()
	if lines.len == 0 {
		return
	}
	parts := lines[0].split(' ')
	if parts.len < 2 {
		return
	}
	method := parts[0]
	path := parts[1]
	if method == 'GET' && path == '/v1/models' {
		send_json(mut conn, 200, models_json)
	} else if method == 'POST' && path == '/v1/chat/completions' {
		chat := json2.decode[ChatRequest](body) or {
			send_json(mut conn, 400, '{"error":"bad chat request"}')
			return
		}
		msgs := chat.messages.map(infer.ChatMsg{
			role:    it.role
			content: it.content
		})
		prompt := infer.apply_chat_template(msgs)
		text := model.generate_stub(prompt, chat.max_tokens)
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
			send_sse(mut conn, id, chat.model, escaped, created)
		} else {
			send_json(mut conn, 200, sb.str())
		}
	} else if method == 'POST' && path == '/v1/completions' {
		comp := json2.decode[CompletionRequest](body) or {
			send_json(mut conn, 400, '{"error":"bad completion request"}')
			return
		}
		text := model.generate_stub(comp.prompt, comp.max_tokens)
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
		send_json(mut conn, 200, sb.str())
	} else {
		send_json(mut conn, 404, '{"error":"not found, try /v1/models or /v1/chat/completions"}')
	}
}

// escape_json escapes one pass (old code chained 3 full-string replaces).
pub fn escape_json(s string) string {
	hex_digits := '0123456789abcdef'
	mut sb := strings.new_builder(s.len + 16)
	for ch in s {
		match ch {
			`\\` { sb.write_string('\\\\') }
			`"` { sb.write_string('\\"') }
			`\n` { sb.write_string('\\n') }
			`\r` { sb.write_string('\\r') }
			`\t` { sb.write_string('\\t') }
			else {
				if ch < 0x20 {
					sb.write_string('\\u00')
					sb.write_u8(hex_digits[ch >> 4])
					sb.write_u8(hex_digits[ch & 0xf])
				} else {
					sb.write_u8(ch)
				}
			}
		}
	}
	return sb.str()
}

fn send_json(mut conn net.TcpConn, code int, payload string) {
	status := if code == 200 {
		'OK'
	} else if code == 400 {
		'Bad Request'
	} else {
		'Not Found'
	}
	mut sb := strings.new_builder(128 + payload.len)
	sb.write_string('HTTP/1.1 ')
	sb.write_string(code.str())
	sb.write_string(' ')
	sb.write_string(status)
	sb.write_string('\r\nContent-Type: application/json\r\nContent-Length: ')
	sb.write_string(payload.len.str())
	sb.write_string('\r\nConnection: close\r\n\r\n')
	sb.write_string(payload)
	conn.write_string(sb.str()) or {}
}

fn send_sse(mut conn net.TcpConn, id string, model string, text string, created i64) {
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
	hb.write_string('HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: ')
	hb.write_string((chunk.len + 18).str())
	hb.write_string('\r\nConnection: close\r\n\r\ndata: ')
	hb.write_string(chunk)
	hb.write_string('\n\ndata: [DONE]\n\n')
	conn.write_string(hb.str()) or {}
}
