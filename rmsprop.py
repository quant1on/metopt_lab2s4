from __future__ import annotations

from typing import Callable

import numpy as np

from logger import CSVLogger
from utils import Bounds, apply_bounds, ensure_float_vector

OptimizationResult = tuple[np.ndarray, float, dict[str, float | int | str]]


def rmsprop(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: Bounds | None = None,
    max_iter: int = 1000,
    tol_grad: float = 1e-6,
    tol_step: float = 1e-8,
    tol_f: float = 1e-12,
    lr: float = 1e-3,
    beta2: float = 0.99,
    eps: float = 1e-8,
    boundary: str = "clip",
    logger: CSVLogger | None = None,
) -> OptimizationResult:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    x = apply_bounds(ensure_float_vector(x0, name="x0"), bounds, boundary)
    f_current = float(f(x))

    x_best = x.copy()
    f_best = f_current

    v = np.zeros_like(x)
    reason = "max_iter"
    iterations = 0

    for iteration in range(max_iter):
        g = ensure_float_vector(grad(x), name="grad")
        grad_norm = float(np.linalg.norm(g))

        v = beta2 * v + (1.0 - beta2) * (g * g)
        step = lr * g / (np.sqrt(v) + eps)
        x_new = apply_bounds(x - step, bounds, boundary)

        step_norm = float(np.linalg.norm(x_new - x))
        f_new = float(f(x_new))

        if f_new < f_best:
            x_best = x_new.copy()
            f_best = f_new

        if logger is not None:
            logger.log(
                iteration=iteration,
                x_best=x_best,
                x_current=x_new,
                f_best=f_best,
                f_current=f_new,
                metric=grad_norm,
            )

        iterations = iteration + 1
        if grad_norm < tol_grad:
            reason = "tol_grad"
            break
        if step_norm < tol_step:
            reason = "tol_step"
            break
        if abs(f_current - f_new) < tol_f:
            reason = "tol_f"
            break

        x = x_new
        f_current = f_new

    meta = {
        "iterations": iterations,
        "reason": reason,
    }
    return x_best, f_best, meta
