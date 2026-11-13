module api

import cfg
import infer

// HandlerCtx carries everything a route handler needs: the loaded model
// plus the resolved configuration. Built once in serve() and shared
// across requests — handlers stay pure per-request logic with no globals.
// New endpoints read ctx; they never touch sockets or process state.
pub struct HandlerCtx {
pub:
	model  &infer.Model
	config cfg.Config
}
