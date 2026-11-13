module api

// OpenAI-compatible JSON shapes. Only the subset vllm-os serves in v1.

pub struct ChatMessage {
pub:
	role    string
	content string
}

pub struct ChatRequest {
pub:
	model       string = 'qwen3-1.7b'
	messages    []ChatMessage
	max_tokens  int = 256
	temperature f32 = 0.7
	top_p       f32 = 0.9
	top_k       int = 40
	seed        u64 // 0 = config seed, config 0 = time-based
	stream      bool
}

pub struct CompletionRequest {
pub:
	model       string = 'qwen3-1.7b'
	prompt      string
	max_tokens  int = 256
	temperature f32 = 0.7
	top_p       f32 = 0.9
	top_k       int = 40
	seed        u64 // 0 = config seed, config 0 = time-based
	stream      bool
}

pub struct ModelCard {
pub:
	id       string
	object   string = 'model'
	owned_by string = 'vllm-os'
}

pub struct ModelsResponse {
pub:
	object string = 'list'
	data   []ModelCard
}
