from __future__ import annotations

from typing import Iterable

import numpy as np

from exceptions import ValidationError

Bounds = tuple[float | np.ndarray, float | np.ndarray]


def ensure_float_vector(x: np.ndarray | Iterable[float], name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be a 1D vector.")
    return arr


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def normalize_bounds(bounds: Bounds | None, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        lower = np.full(dim, -np.inf, dtype=float)
        upper = np.full(dim, np.inf, dtype=float)
        return lower, upper

    if len(bounds) != 2:
        raise ValidationError("bounds must contain (lower, upper).")

    lower_raw, upper_raw = bounds
    lower = np.asarray(lower_raw, dtype=float)
    upper = np.asarray(upper_raw, dtype=float)

    if lower.ndim == 0:
        lower = np.full(dim, float(lower), dtype=float)
    if upper.ndim == 0:
        upper = np.full(dim, float(upper), dtype=float)

    if lower.shape != (dim,) or upper.shape != (dim,):
        raise ValidationError("bounds must be scalars or vectors matching problem dimension.")

    if np.any(lower >= upper):
        raise ValidationError("Each lower bound must be strictly less than upper bound.")

    return lower, upper


def apply_bounds(
    x: np.ndarray,
    bounds: Bounds | None,
    strategy: str = "clip",
) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    lower, upper = normalize_bounds(bounds, x_vec.size)

    if strategy == "clip":
        return np.clip(x_vec, lower, upper)

    if strategy != "reflect":
        raise ValidationError("strategy must be either 'clip' or 'reflect'.")

    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        return np.clip(x_vec, lower, upper)

    span = upper - lower
    reflected = np.mod(x_vec - lower, 2.0 * span)
    reflected = np.where(reflected <= span, reflected, 2.0 * span - reflected)
    return lower + reflected


def make_uniform_point(
    rng: np.random.Generator,
    dim: int,
    bounds: Bounds | None,
) -> np.ndarray:
    lower, upper = normalize_bounds(bounds, dim)
    if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
        return rng.uniform(lower, upper)
    return rng.normal(0.0, 1.0, size=dim)


def make_uniform_population(
    rng: np.random.Generator,
    size: int,
    dim: int,
    bounds: Bounds | None,
) -> list[np.ndarray]:
    if size <= 0:
        raise ValidationError("Population size must be positive.")
    return [make_uniform_point(rng, dim, bounds) for _ in range(size)]
