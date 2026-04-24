from __future__ import annotations

from typing import Callable

import numpy as np

from logger import CSVLogger
from utils import Bounds, apply_bounds, ensure_float_vector, make_rng

OptimizationResult = tuple[np.ndarray, float, dict[str, float | int | str]]


def _fitness_values(
    f: Callable[[np.ndarray], float],
    population: list[np.ndarray],
) -> np.ndarray:
    return np.asarray([float(f(ind)) for ind in population], dtype=float)


def genetic_algorithm(
    f: Callable[[np.ndarray], float],
    x0_population: list[np.ndarray],
    bounds: Bounds | None = None,
    max_iter: int = 200,
    population_size: int | None = None,
    elite_size: int = 4,
    mutation_sigma: float = 0.1,
    crossover_rate: float = 1.0,
    tournament_size: int = 3,
    tol_step: float = 1e-8,
    tol_f: float = 1e-12,
    boundary: str = "clip",
    seed: int | None = None,
    logger: CSVLogger | None = None,
) -> OptimizationResult:
    if not x0_population:
        raise ValueError("x0_population must not be empty.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    rng = make_rng(seed)

    population = [
        apply_bounds(ensure_float_vector(ind, name="population_item"), bounds, boundary)
        for ind in x0_population
    ]

    dim = population[0].size
    if any(ind.size != dim for ind in population):
        raise ValueError("All individuals in x0_population must have the same dimension.")

    if population_size is None:
        population_size = len(population)
    if population_size < 3:
        raise ValueError("population_size must be at least 3.")
    if not (1 <= elite_size < population_size):
        raise ValueError("elite_size must satisfy 1 <= elite_size < population_size.")

    while len(population) < population_size:
        parent = population[rng.integers(0, len(population))]
        mutant = parent + rng.normal(0.0, mutation_sigma, size=dim)
        population.append(apply_bounds(mutant, bounds, boundary))
    population = population[:population_size]

    best_idx = int(np.argmin(_fitness_values(f, population)))
    x_best = population[best_idx].copy()
    f_best = float(f(x_best))

    prev_best = f_best
    reason = "max_iter"
    iterations = 0
    diversity = 0.0

    for iteration in range(max_iter):
        fitness = _fitness_values(f, population)
        order = np.argsort(fitness)
        population = [population[i] for i in order]
        fitness = fitness[order]

        if float(fitness[0]) < f_best:
            x_best = population[0].copy()
            f_best = float(fitness[0])

        population_matrix = np.asarray(population, dtype=float)
        diversity = float(np.mean(np.std(population_matrix, axis=0)))

        if logger is not None:
            logger.log(
                iteration=iteration,
                x_best=x_best,
                x_current=population[0],
                f_best=f_best,
                f_current=float(fitness[0]),
                metric=diversity,
            )

        iterations = iteration + 1
        if diversity < tol_step and abs(prev_best - f_best) < tol_f:
            reason = "tol_step_tol_f"
            break
        prev_best = f_best

        elite = [ind.copy() for ind in population[:elite_size]]

        def select_parent() -> np.ndarray:
            k = min(tournament_size, len(population))
            candidate_idx = rng.choice(len(population), size=k, replace=False)
            best_local = candidate_idx[int(np.argmin(fitness[candidate_idx]))]
            return population[int(best_local)]

        new_population = list(elite)
        while len(new_population) < population_size:
            p1 = select_parent()
            p2 = select_parent()

            if rng.random() < crossover_rate:
                alpha = rng.random(dim)
                child = alpha * p1 + (1.0 - alpha) * p2
            else:
                child = p1.copy() if rng.random() < 0.5 else p2.copy()

            child = child + rng.normal(0.0, mutation_sigma, size=dim)
            child = apply_bounds(child, bounds, boundary)
            new_population.append(child)

        population = new_population

    meta = {
        "iterations": iterations,
        "reason": reason,
        "diversity": diversity,
    }
    return x_best, f_best, meta
