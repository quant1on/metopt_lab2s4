from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

import numpy as np


class CSVLogger:
    _FIELDNAMES = [
        "iteration",
        "x_best",
        "x_current",
        "f_best",
        "f_current",
        "metric",
        "algorithm",
        "function",
        "dim",
    ]

    def __init__(
        self,
        filepath: str | Path,
        algorithm: str,
        function: str,
        dim: int,
    ) -> None:
        self._filepath = Path(filepath)
        self._algorithm = algorithm
        self._function = function
        self._dim = int(dim)
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "CSVLogger":
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filepath, mode="w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDNAMES)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()

    def log(
        self,
        iteration: int,
        x_best: np.ndarray,
        x_current: np.ndarray,
        f_best: float,
        f_current: float,
        metric: float,
    ) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("CSVLogger must be used as a context manager.")

        row = {
            "iteration": int(iteration),
            "x_best": json.dumps(np.asarray(x_best, dtype=float).tolist()),
            "x_current": json.dumps(np.asarray(x_current, dtype=float).tolist()),
            "f_best": float(f_best),
            "f_current": float(f_current),
            "metric": float(metric),
            "algorithm": self._algorithm,
            "function": self._function,
            "dim": self._dim,
        }
        self._writer.writerow(row)
        self._file.flush()
