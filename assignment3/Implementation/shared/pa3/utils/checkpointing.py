"""Periodic checkpointing + resume helpers.

A checkpoint captures everything needed to resume training byte-for-byte:
  * agent (actor / critic / target / log_alpha + optimizers)
  * env step counter
  * RNG states (python, numpy, torch CPU + cuda + mps)
  * replay buffer state (optional — large; off by default)

Files are written to ``<logdir>/checkpoints/step_<XXXXXXXX>.pt``.
``find_latest`` returns the most-recent checkpoint in a directory.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | os.PathLike,
    *,
    agent_state: dict[str, Any],
    step: int,
    buffer_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "agent": agent_state,
        "step": int(step),
        "rng": _rng_state(),
        "buffer": buffer_state,
        "extra": extra or {},
    }
    torch.save(blob, path)


def load_checkpoint(path: str | os.PathLike, map_location: str = "cpu") -> dict[str, Any]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if "rng" in blob:
        _restore_rng(blob["rng"])
    return blob


def find_latest(checkpoint_dir: str | os.PathLike) -> Path | None:
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    files = sorted(d.glob("step_*.pt"))
    return files[-1] if files else None
