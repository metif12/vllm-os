// SPDX-License-Identifier: GPL-2.0-or-later
module krandom

// A volatile field qualifies the pointed-to byte, rather than just the pointer.
struct SecretByte {
mut:
	volatile value u8
}

// OpenBSD's explicit_bzero(3): the stores must survive even when the caller
// never reads the secret again. Keep the C ABI for the networking bridge.
@[export: 'vinix_explicit_bzero']
pub fn explicit_bzero(buf voidptr, len u32) {
	for i := u32(0); i < len; i++ {
		// NOTE(vllm-os): the address math must stay pointer-width. The
		// original `uint(buf)` (32-bit, like the `u32` sweep) truncates
		// high-half addresses (stack, direct map) and faults.
		unsafe { (&SecretByte(u64(buf) + i)).value = 0 }
	}
}
