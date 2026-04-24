from adam import adam
from adamw import adamw
from ga import genetic_algorithm
from gd_momentum import gradient_descent_momentum
from lion import lion
from rmsprop import rmsprop
from sa import simulated_annealing

ALL_OPTIMIZERS = {
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
    "gd_momentum": gradient_descent_momentum,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamw": adamw,
    "lion": lion,
}

GRADIENT_OPTIMIZERS = {
    "gd_momentum",
    "rmsprop",
    "adam",
    "adamw",
    "lion",
}
