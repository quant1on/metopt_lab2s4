from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from exceptions import OptimizationError, ValidationError
from logger import CSVLogger
from utils import ensure_float_vector

OptimizationResult = tuple[np.ndarray, float, dict[str, Any]]


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def _clone_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, list):
            cloned[key] = [item.copy() if isinstance(item, np.ndarray) else item for item in value]
        else:
            cloned[key] = value
    return cloned


def _infer_dim(kwargs: dict[str, Any], fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    if "x0" in kwargs:
        return int(ensure_float_vector(kwargs["x0"], name="x0").size)
    if "x0_population" in kwargs and kwargs["x0_population"]:
        first = ensure_float_vector(kwargs["x0_population"][0], name="x0_population[0]")
        return int(first.size)
    raise ValidationError("Cannot infer dim: provide dim or x0/x0_population.")


def _unpack_result(result: Any) -> OptimizationResult:
    if not isinstance(result, tuple):
        raise OptimizationError("Algorithm must return a tuple.")

    if len(result) == 2:
        x_best, f_best = result
        meta: dict[str, Any] = {}
    elif len(result) == 3:
        x_best, f_best, meta = result
        if not isinstance(meta, dict):
            raise OptimizationError("Third element of result tuple must be a dict.")
    else:
        raise OptimizationError("Algorithm result tuple must have length 2 or 3.")

    x_best_vec = ensure_float_vector(x_best, name="x_best")
    f_best_float = float(f_best)
    if not np.isfinite(f_best_float):
        raise OptimizationError("Algorithm returned non-finite f_best.")

    return x_best_vec, f_best_float, meta


def run_optimization(
    algorithm: Callable[..., Any],
    algo_kwargs: dict[str, Any],
    max_restarts: int = 1,
    log_dir: str | Path | None = None,
    algorithm_name: str = "algorithm",
    function_name: str = "objective",
    dim: int | None = None,
) -> OptimizationResult:
    if max_restarts <= 0:
        raise ValidationError("max_restarts must be positive.")

    algo_name_safe = _sanitize_name(algorithm_name)
    func_name_safe = _sanitize_name(function_name)

    last_error: Exception | None = None

    for attempt in range(1, max_restarts + 1):
        kwargs = _clone_kwargs(algo_kwargs)

        if "seed" in kwargs and kwargs["seed"] is not None:
            kwargs["seed"] = int(kwargs["seed"]) + (attempt - 1)

        run_dim = _infer_dim(kwargs, fallback=dim)

        try:
            if log_dir is not None:
                path = Path(log_dir) / f"{algo_name_safe}_{func_name_safe}_log_attempt{attempt}.csv"
                with CSVLogger(
                    filepath=path,
                    algorithm=algorithm_name,
                    function=function_name,
                    dim=run_dim,
                ) as logger:
                    kwargs["logger"] = logger
                    result = algorithm(**kwargs)
            else:
                result = algorithm(**kwargs)

            x_best, f_best, meta = _unpack_result(result)
            meta = dict(meta)
            meta.setdefault("attempt", attempt)
            meta.setdefault("algorithm", algorithm_name)
            return x_best, f_best, meta

        except (ValidationError, OptimizationError, FloatingPointError, ValueError, TypeError) as exc:
            last_error = exc

    msg = f"Failed after {max_restarts} attempts for {algorithm_name} on {function_name}."
    if last_error is None:
        raise OptimizationError(msg)
    raise OptimizationError(msg) from last_error
