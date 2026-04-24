from __future__ import annotations

from pathlib import Path

import numpy as np

from objectives import ObjectiveSpec, get_objective
from optimizers import ALL_OPTIMIZERS, GRADIENT_OPTIMIZERS
from runner import run_optimization
from utils import make_rng, make_uniform_point, make_uniform_population


def _build_kwargs(name: str, spec: ObjectiveSpec, seed: int) -> dict:
    rng = make_rng(seed)

    if name == "sa":
        return {
            "f": spec.func,
            "x0": make_uniform_point(rng, spec.dim, spec.bounds),
            "bounds": spec.bounds,
            "max_iter": 400,
            "seed": seed,
            "init_temp": 6.0,
            "cooling": 0.99,
            "step_scale": 0.2,
            "boundary": "reflect",
        }

    if name == "ga":
        return {
            "f": spec.func,
            "x0_population": make_uniform_population(rng, size=32, dim=spec.dim, bounds=spec.bounds),
            "bounds": spec.bounds,
            "max_iter": 120,
            "population_size": 32,
            "elite_size": 4,
            "mutation_sigma": 0.12,
            "seed": seed,
            "boundary": "reflect",
        }

    x0 = make_uniform_point(rng, spec.dim, spec.bounds)
    common = {
        "f": spec.func,
        "grad": spec.grad,
        "x0": x0,
        "bounds": spec.bounds,
        "max_iter": 700,
        "tol_grad": 1e-7,
    }

    if name == "gd_momentum":
        common.update({"lr": 2e-3, "beta": 0.9, "boundary": "clip"})
    elif name == "rmsprop":
        common.update({"lr": 4e-3, "beta2": 0.99, "eps": 1e-8, "boundary": "clip"})
    elif name == "adam":
        common.update({"lr": 8e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "boundary": "clip"})
    elif name == "adamw":
        common.update(
            {
                "lr": 8e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 1e-3,
                "boundary": "clip",
            }
        )
    elif name == "lion":
        common.update(
            {
                "lr": 2e-3,
                "beta1": 0.9,
                "beta2": 0.99,
                "weight_decay": 5e-4,
                "boundary": "clip",
            }
        )
    return common


def run_demo() -> None:
    log_dir = Path(__file__).resolve().parent
    tasks = [
        ("rosenbrock", 2),
        ("rastrigin", 5),
        ("levi_n13", 2),
    ]

    results: list[dict[str, float | int | str]] = []

    for function_name, dim in tasks:
        spec = get_objective(function_name, dim=dim)
        print(f"\n=== {spec.name} (dim={spec.dim}) ===")

        for idx, (algorithm_name, algorithm) in enumerate(ALL_OPTIMIZERS.items()):
            if algorithm_name in GRADIENT_OPTIMIZERS and function_name == "levi_n13" and dim != 2:
                continue

            kwargs = _build_kwargs(algorithm_name, spec, seed=100 + idx)
            x_best, f_best, meta = run_optimization(
                algorithm=algorithm,
                algo_kwargs=kwargs,
                max_restarts=1,
                log_dir=log_dir,
                algorithm_name=algorithm_name,
                function_name=spec.name,
                dim=spec.dim,
            )

            results.append(
                {
                    "algorithm": algorithm_name,
                    "function": spec.name,
                    "dim": spec.dim,
                    "f_best": f_best,
                    "iters": int(meta.get("iterations", 0)),
                    "reason": str(meta.get("reason", "n/a")),
                }
            )
            print(f"{algorithm_name:12s} f_best={f_best:12.6f} iters={meta.get('iterations', 0):4}")

    print("\n=== Summary ===")
    for row in results:
        print(
            f"{row['algorithm']:12s} | {row['function']:10s} | dim={row['dim']:2d} | "
            f"f_best={row['f_best']:12.6f} | iters={row['iters']:4d} | {row['reason']}"
        )


if __name__ == "__main__":
    run_demo()
