// SPDX-License-Identifier: GPL-2.0-or-later
module resource

import stat

// Network resources may need an RPC before returning attributes. The backend
// returns the value while its lock is held, so a concurrent refresh cannot
// change individual fields between refresh and the caller's copy.
pub interface StatSnapshotResource {
mut:
	snapshot_stat() ?stat.Stat
}

struct StatSnapshotCall {
mut:
	backend StatSnapshotResource
}

pub fn snapshot_stat(mut res Resource) ?stat.Stat {
	// NOTE(vllm-os): `is StatSnapshotResource` intentionally not used:
	// nothing implements it and the pinned V compiles that `is` as a
	// non-nil check (see fs/tmpfs.v), which would dispatch-panic below.
	return res.stat
}
