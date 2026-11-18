module main

import compose
import os
import time
import ui2

// Thin ui2 shell over compose.validate_iso_form / build_iso_command /
// boot_command (gui/compose, unit-tested). Controls are read fresh on
// Generate; outputs are written back by id. The Generate button is the
// only logic entry point besides the toggle, so verifying the composer
// plus a control-rendering smoke test covers the app.
//
// Toolchain: this shell needs a json2-capable V (ui2's working copy
// imports json2, absent from the 0.5.2 release). Repo core stays pinned
// (see docs/development.md); build ONLY this dir with a weekly V, e.g.
// D:\MyProjects\v\v.exe -o build_iso_ui.exe gui. Never `v fmt -w`
// the repo with that binary.
@[heap]
struct BuilderUI {
mut:
	stub     bool
	built    bool
	verified bool
	error    string
}

const builder_ui = &BuilderUI{}
const win_w = 660
const win_h = 900
const col_w = 628

fn lab(id string, y f64, text string) ui2.Element {
	return ui2.label(id, text, ui2.rect(16, y, col_w, 20), ui2.TextStyle{ size: 13 })
}

fn field(id string, y f64, initial string) ui2.Element {
	return ui2.text_field_with_change_and_submit(id, id + '-submit', '', initial, ui2.rect(16, y,
		col_w, 32), ui2.BoxStyle{}, ui2.TextStyle{}, ui2.keyboard_default)
}

fn pair_label(id string, x f64, y f64, text string) ui2.Element {
	return ui2.label(id, text, ui2.rect(x, y, 296, 20), ui2.TextStyle{ size: 13 })
}

fn pair_field(id string, x f64, y f64, initial string) ui2.Element {
	return ui2.text_field_with_change_and_submit(id, id + '-submit', '', initial, ui2.rect(x, y,
		296, 32), ui2.BoxStyle{}, ui2.TextStyle{}, ui2.keyboard_default)
}

fn pair_dropdown(id string, x f64, y f64, selected string, options []string) ui2.Element {
	return ui2.dropdown(id, selected, options, ui2.rect(x, y, 296, 32), ui2.BoxStyle{}, ui2.TextStyle{})
}

fn build_builder_screen() ui2.Element {
	mut st := unsafe { builder_ui }
	st.built = true
	stub_title := if st.stub { 'stub tokenizer: yes' } else { 'stub tokenizer: no' }
	return ui2.screen(0xf3f4f6, [
		ui2.label('title', 'vllm-os ISO builder', ui2.rect(16, 12, col_w, 28), ui2.TextStyle{
			size: 18
			bold: true
		}),
		lab('l-arch', 48, 'Target arch (only x86_64 boots)'),
		ui2.dropdown('arch', 'x86_64', ['x86_64', 'arm64'], ui2.rect(16, 70, 296, 32), ui2.BoxStyle{}, ui2.TextStyle{}),
		lab('l-model', 110, 'GGUF weights path (WSL-visible)'),
		field('model', 132, 'models/Qwen3-0.6B-Q4_0.gguf'),
		lab('l-tok', 172, 'Tokenizer path (empty = next to model)'),
		field('tokenizer', 194, ''),
		ui2.button('stub_toggle', stub_title, ui2.rect(16, 234, 220, 34), ui2.BoxStyle{}, ui2.TextStyle{}),
		pair_label('l-port', 274, 274, 'Port'),
		pair_field('port', 274, 296, '8000'),
		pair_label('l-ctx', 348, 274, 'Ctx cap'),
		pair_field('ctx', 348, 296, '4096'),
		pair_label('l-backend', 16, 336, 'Backend'),
		pair_dropdown('backend', 16, 358, 'cpu', ['cpu', 'cpu-blocked',
			'cpu-simd', 'cpu-parallel', 'auto']),
		pair_label('l-preset', 348, 336, 'Sampler preset'),
		pair_dropdown('preset', 348, 358, 'balanced', ['deterministic', 'precise',
			'balanced', 'creative']),
		pair_label('l-temp', 16, 398, 'Temp (empty = default)'),
		pair_field('temp', 16, 420, ''),
		pair_label('l-mem', 348, 398, 'Guest RAM (GiB)'),
		pair_field('mem', 348, 420, '4'),
		lab('l-auth', 460, 'Auth token (empty = open mode)'),
		field('auth', 482, ''),
		ui2.button('generate', 'Generate', ui2.rect(16, 522, 200, 34), ui2.BoxStyle{}, ui2.TextStyle{}),
		ui2.button('copy', 'Copy command', ui2.rect(228, 522, 200, 34), ui2.BoxStyle{}, ui2.TextStyle{}),
		ui2.label('err', '', ui2.rect(16, 562, col_w, 22), ui2.TextStyle{ size: 13 }),
		lab('l-cmd', 588, 'Build command'),
		ui2.text_area('cmd', '', ui2.rect(16, 610, col_w, 76), ui2.BoxStyle{
			bg: 0xf8fafc
		}, ui2.TextStyle{}),
		lab('l-boot', 692, 'Boot'),
		ui2.text_area('boot', '', ui2.rect(16, 714, col_w, 60), ui2.BoxStyle{
			bg: 0xf8fafc
		}, ui2.TextStyle{}),
	])
}

fn read_form() compose.IsoForm {
	st := unsafe { builder_ui }
	return compose.IsoForm{
		arch:      ui2.text('arch')
		model:     ui2.text('model').trim_space()
		tokenizer: ui2.text('tokenizer').trim_space()
		stub:      st.stub
		port:      ui2.text('port').trim_space()
		ctx:       ui2.text('ctx').trim_space()
		backend:   ui2.text('backend')
		preset:    ui2.text('preset')
		temp:      ui2.text('temp').trim_space()
		mem:       ui2.text('mem').trim_space()
		auth:      ui2.text('auth')
	}
}

fn generate_outputs() {
	f := read_form()
	if f.model != '' && !os.exists(f.model) {
		ui2.set_text('err', 'model file not found: ' + f.model)
		ui2.set_text('cmd', '')
		ui2.set_text('boot', '')
		return
	}
	model_bytes := u64(os.file_size(f.model))
	err := compose.validate_iso_form(f, model_bytes)
	ui2.set_text('err', err)
	if err != '' {
		ui2.set_text('cmd', '')
		ui2.set_text('boot', '')
		return
	}
	ui2.set_text('cmd', compose.build_iso_command(f))
	ui2.set_text('boot', compose.boot_command(f))
}

fn handle_builder_event(event string) {
	mut st := unsafe { builder_ui }
	match event {
		'stub_toggle' {
			st.stub = !st.stub
			ui2.refresh()
		}
		'generate' {
			generate_outputs()
		}
		'copy' {
			ui2.set_clipboard_text(ui2.text('cmd'))
		}
		else {}
	}
}

fn smoke_verifier() {
	mut st := unsafe { builder_ui }
	mut last_error := 'window never built'
	for _ in 0 .. 100 {
		if st.built {
			mut checks := {
				'arch':    'x86_64'
				'model':   'models/Qwen3-0.6B-Q4_0.gguf'
				'port':    '8000'
				'ctx':     '4096'
				'backend': 'cpu'
				'preset':  'balanced'
				'mem':     '4'
			}
			mut ready := true
			for id, expected in checks {
				actual := ui2.text(id)
				if actual != expected {
					last_error = '${id}: expected `${expected}`, got `${actual}`'
					ready = false
					break
				}
			}
			if ready {
				st.verified = true
				ui2.quit()
				return
			}
		}
		time.sleep(100 * time.millisecond)
	}
	st.error = last_error
	ui2.quit()
}

	fn main() {
	smoke := '--smoke' in os.args
	if smoke {
		spawn smoke_verifier()
	}
	ui2.run_window('vllm-os ISO builder', win_w, win_h, build_builder_screen, handle_builder_event)
	if smoke {
		st := unsafe { builder_ui }
		// GUI-subsystem binaries have no console: report via file like
		// ui2's own windows_smoke (BUILDER_SMOKE_RESULT), stdout as fallback.
		result := os.getenv('BUILDER_SMOKE_RESULT')
		if !st.built || !st.verified {
			msg := 'BUILDER_SMOKE_FAILED: ' + st.error
			if result != '' {
				os.write_file(result, msg + '\n') or {}
			} else {
				eprintln(msg)
			}
			exit(1)
		}
		if result != '' {
			os.write_file(result, 'BUILDER_SMOKE_OK\n') or { exit(1) }
		} else {
			println('BUILDER_SMOKE_OK')
		}
	}
}
