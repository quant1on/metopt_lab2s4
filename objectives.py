from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from exceptions import ValidationError
from utils import ensure_float_vector

ObjectiveFn = Callable[[np.ndarray], float]
GradientFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    dim: int
    func: ObjectiveFn
    grad: GradientFn
    optimum_point: np.ndarray
    optimum_value: float
    bounds: tuple[float, float]


def rosenbrock_nd(x: np.ndarray) -> float:
    x_vec = ensure_float_vector(x)
    if x_vec.size < 2:
        raise ValidationError("Rosenbrock is defined for dim >= 2.")
    return float(np.sum(100.0 * (x_vec[1:] - x_vec[:-1] ** 2) ** 2 + (1.0 - x_vec[:-1]) ** 2))


def rosenbrock_grad_nd(x: np.ndarray) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    if x_vec.size < 2:
        raise ValidationError("Rosenbrock gradient is defined for dim >= 2.")

    grad = np.zeros_like(x_vec)
    grad[0] = -400.0 * x_vec[0] * (x_vec[1] - x_vec[0] ** 2) + 2.0 * (x_vec[0] - 1.0)
    grad[-1] = 200.0 * (x_vec[-1] - x_vec[-2] ** 2)
    if x_vec.size > 2:
        grad[1:-1] = (
            200.0 * (x_vec[1:-1] - x_vec[:-2] ** 2)
            - 400.0 * x_vec[1:-1] * (x_vec[2:] - x_vec[1:-1] ** 2)
            + 2.0 * (x_vec[1:-1] - 1.0)
        )
    return grad


def rosenbrock_2d(x: np.ndarray) -> float:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Rosenbrock 2D expects dim=2.")
    return rosenbrock_nd(x_vec)


def rosenbrock_grad_2d(x: np.ndarray) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Rosenbrock 2D gradient expects dim=2.")
    return rosenbrock_grad_nd(x_vec)


def rastrigin_nd(x: np.ndarray, a: float = 10.0) -> float:
    x_vec = ensure_float_vector(x)
    n = x_vec.size
    return float(a * n + np.sum(x_vec**2 - a * np.cos(2.0 * np.pi * x_vec)))


def rastrigin_grad_nd(x: np.ndarray, a: float = 10.0) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    return 2.0 * x_vec + 2.0 * np.pi * a * np.sin(2.0 * np.pi * x_vec)


def rastrigin_2d(x: np.ndarray) -> float:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Rastrigin 2D expects dim=2.")
    return rastrigin_nd(x_vec)


def rastrigin_grad_2d(x: np.ndarray) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Rastrigin 2D gradient expects dim=2.")
    return rastrigin_grad_nd(x_vec)


def levi_n13_2d(x: np.ndarray) -> float:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Levi N13 is defined only for dim=2.")

    x1, x2 = x_vec
    term1 = np.sin(3.0 * np.pi * x1) ** 2
    term2 = (x1 - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x2) ** 2)
    term3 = (x2 - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x2) ** 2)
    return float(term1 + term2 + term3)


def levi_n13_grad_2d(x: np.ndarray) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Levi N13 gradient is defined only for dim=2.")

    x1, x2 = x_vec

    dfdx1 = (
        3.0 * np.pi * np.sin(6.0 * np.pi * x1)
        + 2.0 * (x1 - 1.0) * (1.0 + np.sin(3.0 * np.pi * x2) ** 2)
    )

    dfdx2 = (
        3.0 * np.pi * (x1 - 1.0) ** 2 * np.sin(6.0 * np.pi * x2)
        + 2.0 * (x2 - 1.0) * (1.0 + np.sin(2.0 * np.pi * x2) ** 2)
        + 2.0 * np.pi * (x2 - 1.0) ** 2 * np.sin(4.0 * np.pi * x2)
    )

    return np.array([dfdx1, dfdx2], dtype=float)

def stepped_himmelblau_2d(x: np.ndarray, d: float = 0.18) -> float:
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Stepped Himmelblau 2D expects dim=2.")

    x1, x2 = x_vec
    
    c1 = np.round(np.sin(10.0 * x2)) + 2.0
    c2 = np.round(np.sin(7.0 * x1)) + 2.0
    
    term1 = ((x1 * c1) ** 2 + x2 - 10.0) ** 2
    term2 = (x1 + (x2 * c2) ** 2 - 7.0) ** 2
    
    return float(d * (term1 + term2))


def stepped_himmelblau_grad_2d(x: np.ndarray, d: float = 0.18) -> np.ndarray:
    """
    Локальный градиент для модифицированной функции Химмельблау.
    Внимание: производная от round() игнорируется (принимается за 0 локально).
    """
    x_vec = ensure_float_vector(x)
    if x_vec.size != 2:
        raise ValidationError("Stepped Himmelblau 2D gradient expects dim=2.")

    x1, x2 = x_vec
    
    c1 = np.round(np.sin(10.0 * x2)) + 2.0
    c2 = np.round(np.sin(7.0 * x1)) + 2.0
    
    part1 = (x1 * c1) ** 2 + x2 - 10.0
    part2 = x1 + (x2 * c2) ** 2 - 7.0
    
    dfdx1 = d * (2.0 * part1 * 2.0 * x1 * (c1 ** 2) + 2.0 * part2)
    dfdx2 = d * (2.0 * part1 + 2.0 * part2 * 2.0 * x2 * (c2 ** 2))

    return np.array([dfdx1, dfdx2], dtype=float)

def get_objective(name: str, dim: int = 2) -> ObjectiveSpec:
    key = name.strip().lower()

    if key == "rosenbrock":
        if dim < 2:
            raise ValidationError("Rosenbrock objective expects dim >= 2.")
        return ObjectiveSpec(
            name="rosenbrock",
            dim=dim,
            func=rosenbrock_nd,
            grad=rosenbrock_grad_nd,
            optimum_point=np.ones(dim, dtype=float),
            optimum_value=0.0,
            bounds=(-3.0, 3.0),
        )

    if key == "rastrigin":
        if dim < 2:
            raise ValidationError("Rastrigin objective expects dim >= 2.")
        return ObjectiveSpec(
            name="rastrigin",
            dim=dim,
            func=rastrigin_nd,
            grad=rastrigin_grad_nd,
            optimum_point=np.zeros(dim, dtype=float),
            optimum_value=0.0,
            bounds=(-5.12, 5.12),
        )

    if key in {"levi", "levi_n13", "levi-n13"}:
        if dim != 2:
            raise ValidationError("Levi N13 is available only in 2D.")
        return ObjectiveSpec(
            name="levi_n13",
            dim=2,
            func=levi_n13_2d,
            grad=levi_n13_grad_2d,
            optimum_point=np.array([1.0, 1.0], dtype=float),
            optimum_value=0.0,
            bounds=(-10.0, 10.0),
        )

    raise ValidationError(f"Unknown objective: {name}")


def evaluate_objective(name: str, x: np.ndarray) -> float:
    x_vec = ensure_float_vector(x)
    spec = get_objective(name=name, dim=x_vec.size)
    return spec.func(x_vec)


def evaluate_gradient(name: str, x: np.ndarray) -> np.ndarray:
    x_vec = ensure_float_vector(x)
    spec = get_objective(name=name, dim=x_vec.size)
    return spec.grad(x_vec)
