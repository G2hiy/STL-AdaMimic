"""Task-specific STL specifications for 创新点②.

Each task spec exposes a `build(cfg, env_ctx) -> STLContext` factory that returns
a compiled STL spec rooted at `STLContext.spec_root`, a list of accumulators
that the env must drive every step (via `acc.step()`) and reset per episode
(via `acc.reset(env_ids)`).
"""
from __future__ import annotations

from typing import Callable, Dict

from . import far_jump

TASK_SPECS: Dict[str, Callable] = {
    "far_jump": far_jump.build,
}

__all__ = ["TASK_SPECS", "far_jump"]
