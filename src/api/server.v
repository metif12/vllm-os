module api

import json2
import net
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

// serve starts the OpenAI-compatible HTTP API on the given port.
pub fn serve(port int, model &infer.Model) ! {
	mut ln := net.listen_tcp(.ip6, ':${port}')!
	eprintln('[vllm-os] openai-compat api on :${port} model=${model.cfg.model_id} quant=${model.cfg.quant}')
	for {
		mut conn := ln.accept()!
		spawn handle(mut conn, model)
	}
}

fn handle(mut conn net.TcpConn, model &infer.Model) {
	defer {
		conn.close() or {}
	}
	mut buf := []u8{len: 65536}
	n := conn.read(mut buf) or { return }
	if n <= 0 {
		return
	}
	req := buf[..n].bytestr()
	head_end := req.index('\r\n\r\n') or { return }
	head := req[..head_end]
	body := if head_end + 4 < req.len { req[head_end + 4..] } else { '' }
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
		resp := ModelsResponse{
			data: [ModelCard{
				id: 'qwen3-1.7b'
			}]
		}
		send_json(mut conn, 200, json2.encode(resp, escape_unicode: true, time_as_unix: true))
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
		escaped := text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
		payload := '{"id":"${id}","object":"chat.completion","created":${created},"model":"${chat.model}","choices":[{"index":0,"message":{"role":"assistant","content":"${escaped}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0}}'
		if chat.stream {
			send_sse(mut conn, id, chat.model, escaped, created)
		} else {
			send_json(mut conn, 200, payload)
		}
	} else if method == 'POST' && path == '/v1/completions' {
		comp := json2.decode[CompletionRequest](body) or {
			send_json(mut conn, 400, '{"error":"bad completion request"}')
			return
		}
		text := model.generate_stub(comp.prompt, comp.max_tokens)
		id := 'cmpl-${time.now().unix_nano()}'
		created := time.now().unix()
		escaped := text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
		payload := '{"id":"${id}","object":"text_completion","created":${created},"model":"${comp.model}","choices":[{"text":"${escaped}","index":0,"finish_reason":"stop"}]}'
		send_json(mut conn, 200, payload)
	} else {
		send_json(mut conn, 404, '{"error":"not found, try /v1/models or /v1/chat/completions"}')
	}
}

fn send_json(mut conn net.TcpConn, code int, payload string) {
	status := if code == 200 {
		'OK'
	} else if code == 400 {
		'Bad Request'
	} else {
		'Not Found'
	}
	res := 'HTTP/1.1 ${code} ${status}\r\nContent-Type: application/json\r\nContent-Length: ${payload.len}\r\nConnection: close\r\n\r\n${payload}'
	conn.write_string(res) or {}
}

fn send_sse(mut conn net.TcpConn, id string, model string, text string, created i64) {
	chunk := '{"id":"${id}","object":"chat.completion.chunk","created":${created},"model":"${model}","choices":[{"index":0,"delta":{"content":"${text}"},"finish_reason":null}]}'
	body := 'data: ${chunk}\n\ndata: [DONE]\n\n'
	res := 'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: ${body.len}\r\nConnection: close\r\n\r\n${body}'
	conn.write_string(res) or {}
}
