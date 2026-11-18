module compose

fn test_validate_happy_path() {
	f := IsoForm{
		model:     'models/Qwen3-0.6B-Q4_0.gguf'
		tokenizer: 'models/tokenizer.json'
		preset:    'precise'
		temp:      '0.5'
	}
	assert validate_iso_form(f, 100) == ''
	cmd := build_iso_command(f)
	assert cmd.contains('--model models/Qwen3-0.6B-Q4_0.gguf')
	assert cmd.contains('--tokenizer models/tokenizer.json')
	assert cmd.contains('--port 8000')
	assert cmd.contains('--ctx 4096')
	assert cmd.contains('--preset precise')
	assert cmd.contains('--temp 0.5')
	assert cmd.contains('--mem 4')
	assert !cmd.contains('--allow-stub-tokenizer')
	assert !cmd.contains('--auth-token')
	boot := boot_command(f)
	assert boot.contains('hostfwd=tcp::18000-:8000')
	assert boot.contains('-m 4G')
}

fn test_validate_refusals() {
	mut f := IsoForm{}
	assert validate_iso_form(f, 0) == 'model path required'
	f = IsoForm{arch: 'arm64', model: 'm.gguf'}
	assert validate_iso_form(f, 0).starts_with('refused:')
	f = IsoForm{model: 'm.gguf', port: 'abc'}
	assert validate_iso_form(f, 0).contains('port')
	f = IsoForm{model: 'm.gguf', port: '0'}
	assert validate_iso_form(f, 0).contains('port')
	f = IsoForm{model: 'm.gguf', ctx: '10'}
	assert validate_iso_form(f, 0).contains('ctx')
	f = IsoForm{model: 'm.gguf', backend: 'wat'}
	assert validate_iso_form(f, 0).contains('backend')
	f = IsoForm{model: 'm.gguf', preset: 'wat'}
	assert validate_iso_form(f, 0).contains('preset')
	f = IsoForm{model: 'm.gguf', temp: 'abc'}
	assert validate_iso_form(f, 0).contains('temp')
	f = IsoForm{model: 'm.gguf', temp: '9'}
	assert validate_iso_form(f, 0).contains('temp')
	// 2 GiB file in a 1 GiB guest (plus 1 GiB runtime headroom).
	f = IsoForm{model: 'm.gguf', mem: '1'}
	assert validate_iso_form(f, 2 * 1073741824) == 'weights + 1 GiB runtime exceed the guest budget'
}

fn test_validate_stub_and_auth() {
	f := IsoForm{
		model: 'm.gguf'
		stub:  true
		auth:  "s3'cret"
	}
	assert validate_iso_form(f, 0) == ''
	cmd := build_iso_command(f)
	assert cmd.contains('--allow-stub-tokenizer')
	assert cmd.contains("--auth-token 's3'\\''cret'")
	assert !cmd.contains('--tokenizer')
}
