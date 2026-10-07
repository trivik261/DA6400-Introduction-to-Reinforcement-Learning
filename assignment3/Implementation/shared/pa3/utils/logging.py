"""Lightweight CSV logger keyed by environment timesteps."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict


class CSVLogger:
    """Append-only CSV logger.  One row per `log()` call.

    Columns are inferred from the first call.  Subsequent calls must use the
    same set of keys (extra keys are ignored, missing ones written as empty).
    """

    def __init__(self, path: str | os.PathLike, flush_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._flush_every = flush_every
        self._n = 0
        self._t0 = time.time()

    def log(self, row: Dict[str, float]) -> None:
        if self._writer is None:
            self._fields = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k, "") for k in self._fields})
        self._n += 1
        if self._n % self._flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
