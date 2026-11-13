module api

// Route table: the single place endpoints are registered. Adding an
// endpoint = one row here + one handler fn in handlers.v + its JSON
// shapes in types.v. No socket code changes, no dispatch edits.

// HandlerResult is what a handler returns; server.v frames it onto TCP.
pub struct HandlerResult {
pub:
	status       int
	content_type string = 'application/json'
	body         string
}

// HandlerFn handles one request body with shared ctx (see context.v).
pub type HandlerFn = fn (body string, ctx &HandlerCtx) HandlerResult

pub struct Route {
pub:
	method      string
	path        string
	description string
	handler     ?HandlerFn
}

// routes lists every endpoint. Order matters only for the docs table.
pub fn routes() []Route {
	return [
		Route{'GET', '/health', 'liveness + model id', handle_health},
		Route{'GET', '/v1/models', 'list models (OpenAI shape)', handle_models},
		Route{'POST', '/v1/chat/completions', 'chat completion, stream:true gives SSE', handle_chat},
		Route{'POST', '/v1/completions', 'legacy completion', handle_completions},
	]
}

// match_route finds the route for method+path, or none for 404s.
// Pure function — unit-tested, no I/O.
pub fn match_route(method string, path string) ?Route {
	for r in routes() {
		if r.method == method && r.path == path {
			return r
		}
	}
	return none
}
