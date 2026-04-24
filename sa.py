from __future__ import annotations

import math
from typing import Callable

import numpy as np

from logger import CSVLogger
from utils import Bounds, apply_bounds, ensure_float_vector, make_rng

OptimizationResult = tuple[np.ndarray, float, dict[str, float | int | str]]


def simulated_annealing(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Bounds | None = None,
    max_iter: int = 2000,
    tol_step: float = 1e-8,
    tol_f: float = 1e-12,
    init_temp: float = 5.0,
    cooling: float = 0.995,
    step_scale: float = 0.25,
    boundary: str = "clip",
    seed: int | None = None,
    logger: CSVLogger | None = None,
) -> OptimizationResult:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if init_temp <= 0.0:
        raise ValueError("init_temp must be positive.")
    if not (0.0 < cooling < 1.0):
        raise ValueError("cooling must be in (0, 1).")

    rng = make_rng(seed)

    x_current = apply_bounds(ensure_float_vector(x0, name="x0"), bounds, boundary)
    f_current = float(f(x_current))

    x_best = x_current.copy()
    f_best = f_current

    temp = float(init_temp)
    prev_best = f_best
    stall_count = 0
    patience = max(50, min(500, max_iter // 5))
    reason = "max_iter"
    iterations = 0

    for iteration in range(max_iter):
        step = rng.normal(0.0, step_scale, size=x_current.size)
        x_candidate = apply_bounds(x_current + step, bounds, boundary)
        f_candidate = float(f(x_candidate))

        delta = f_candidate - f_current
        accept = delta <= 0.0
        if not accept:
            accept_prob = math.exp(-delta / max(temp, 1e-12))
            accept = rng.random() < accept_prob

        if accept:
            x_current = x_candidate
            f_current = f_candidate

        if f_current < f_best:
            x_best = x_current.copy()
            f_best = f_current

        if logger is not None:
            logger.log(
                iteration=iteration,
                x_best=x_best,
                x_current=x_current,
                f_best=f_best,
                f_current=f_current,
                metric=temp,
            )

        iterations = iteration + 1
        proposal_norm = float(np.linalg.norm(step))

        improved = (prev_best - f_best) > tol_f
        if improved:
            stall_count = 0
        else:
            stall_count += 1

        # Do not stop on rejected proposals; stop only when both step scale and
        # temperature are effectively exhausted.
        if proposal_norm < tol_step and temp < 1e-6:
            reason = "tol_step"
            break
        if stall_count >= patience:
            reason = "tol_f"
            break

        prev_best = f_best
        temp *= cooling

    meta = {
        "iterations": iterations,
        "reason": reason,
        "final_temperature": temp,
    }
    return x_best, f_best, meta
