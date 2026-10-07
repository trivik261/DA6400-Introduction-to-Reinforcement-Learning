"""Set sensible torch thread counts.

For the small MLPs used in this assignment, PyTorch's default of "all cores"
causes massive thread-context-switch overhead on macOS — training slows down by
~20×.  Capping torch threads to 2 is a good default, and lets us run multiple
seeds in parallel without thrashing.

Override with the env var ``PA3_TORCH_THREADS``.
"""
from __future__ import annotations

import os

import torch


def configure_threads(default: int = 2) -> int:
    n = int(os.environ.get("PA3_TORCH_THREADS", default))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except RuntimeError:
        pass
    return n
