from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from adam import adam
from adamw import adamw
from ga import genetic_algorithm
from gd_momentum import gradient_descent_momentum
from lion import lion
from objectives import get_objective
from rmsprop import rmsprop
from runner import run_optimization
from sa import simulated_annealing
from utils import make_rng, make_uniform_population


EXPECTED_COLUMNS = {
    "iteration",
    "x_best",
    "x_current",
    "f_best",
    "f_current",
    "metric",
    "algorithm",
    "function",
    "dim",
}


def finite_diff_grad(
    f,
    x: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        g[i] = (f(x_plus) - f(x_minus)) / (2.0 * h)
    return g


@pytest.mark.parametrize(
    "name, dim",
    [
        ("rosenbrock", 2),
        ("rosenbrock", 5),
        ("rastrigin", 2),
        ("rastrigin", 10),
        ("levi_n13", 2),
    ],
)
def test_objective_minimum_is_zero(name: str, dim: int) -> None:
    spec = get_objective(name, dim=dim)
    value = spec.func(spec.optimum_point)
    assert value == pytest.approx(spec.optimum_value, abs=1e-12)


@pytest.mark.parametrize(
    "name, dim, point",
    [
        ("rosenbrock", 3, np.array([1.2, 0.8, 1.1], dtype=float)),
        ("rastrigin", 4, np.array([0.2, -0.4, 1.1, -1.0], dtype=float)),
        ("levi_n13", 2, np.array([0.4, 1.6], dtype=float)),
    ],
)
def test_analytic_gradient_matches_numeric(name: str, dim: int, point: np.ndarray) -> None:
    spec = get_objective(name, dim=dim)
    g_analytic = spec.grad(point)
    g_numeric = finite_diff_grad(spec.func, point)
    assert np.allclose(g_analytic, g_numeric, rtol=1e-4, atol=1e-4)


def test_sa_smoke_improves() -> None:
    spec = get_objective("rosenbrock", dim=2)
    x0 = np.array([-1.4, 1.4], dtype=float)
    f0 = spec.func(x0)

    x_best, f_best, _ = simulated_annealing(
        f=spec.func,
        x0=x0,
        bounds=spec.bounds,
        max_iter=400,
        init_temp=6.0,
        cooling=0.99,
        step_scale=0.2,
        boundary="reflect",
        seed=123,
    )

    assert np.all(np.isfinite(x_best))
    assert math.isfinite(f_best)
    assert f_best < f0


def test_ga_smoke_improves() -> None:
    spec = get_objective("rosenbrock", dim=2)
    rng = make_rng(123)
    population = make_uniform_population(rng, size=28, dim=2, bounds=spec.bounds)
    f0 = min(spec.func(ind) for ind in population)

    x_best, f_best, _ = genetic_algorithm(
        f=spec.func,
        x0_population=population,
        bounds=spec.bounds,
        max_iter=80,
        population_size=28,
        elite_size=4,
        mutation_sigma=0.1,
        boundary="reflect",
        seed=123,
    )

    assert np.all(np.isfinite(x_best))
    assert math.isfinite(f_best)
    assert f_best < f0


@pytest.mark.parametrize(
    "method, kwargs",
    [
        (gradient_descent_momentum, {"lr": 2e-3, "beta": 0.9}),
        (rmsprop, {"lr": 4e-3, "beta2": 0.99, "eps": 1e-8}),
        (adam, {"lr": 8e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
        (
            adamw,
            {
                "lr": 8e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 1e-3,
            },
        ),
        (
            lion,
            {
                "lr": 2e-3,
                "beta1": 0.9,
                "beta2": 0.99,
                "weight_decay": 5e-4,
            },
        ),
    ],
)
def test_gradient_methods_smoke(method, kwargs) -> None:
    spec = get_objective("rosenbrock", dim=2)
    x0 = np.array([-1.3, 1.2], dtype=float)
    f0 = spec.func(x0)

    x_best, f_best, meta = method(
        f=spec.func,
        grad=spec.grad,
        x0=x0,
        bounds=spec.bounds,
        max_iter=700,
        tol_grad=1e-7,
        boundary="clip",
        **kwargs,
    )

    assert np.all(np.isfinite(x_best))
    assert math.isfinite(f_best)
    assert f_best < f0
    assert int(meta.get("iterations", 0)) > 0


def test_runner_creates_csv_log(tmp_path: Path) -> None:
    spec = get_objective("rosenbrock", dim=2)
    x0 = np.array([-1.2, 1.0], dtype=float)

    _, f_best, _ = run_optimization(
        algorithm=adam,
        algo_kwargs={
            "f": spec.func,
            "grad": spec.grad,
            "x0": x0,
            "bounds": spec.bounds,
            "max_iter": 30,
            "lr": 8e-3,
        },
        max_restarts=1,
        log_dir=tmp_path,
        algorithm_name="adam",
        function_name=spec.name,
        dim=spec.dim,
    )

    assert math.isfinite(f_best)

    log_path = tmp_path / "adam_rosenbrock_log_attempt1.csv"
    assert log_path.exists()

    with open(log_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows
    assert EXPECTED_COLUMNS.issubset(set(reader.fieldnames or []))
