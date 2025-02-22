"""Module which provides source for big test of distribution mixtures of single model"""

import random

import numpy as np

from examples.config import MAX_WORKERS, TESTS_OPTIMIZERS
from examples.mono_test_generator import generate_mono_test
from examples.utils import Clicker, Test, init_solver, run_tests, save_results
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(model: type[AModelWithGenerator], o_borders: list[tuple[float, float]]) -> list[Test]:
        return generate_mono_test(
            model_t=model,
            params_borders=o_borders,
            clicker=counter,
            ks=[1, 2, 3],
            sizes=[50, 100, 200, 500, 1000],
            distributions_count=32,
            base_size=2048,
            tests_per_size=8,
            tests_per_cond=2,
            runs_per_test=1,
            solvers=[init_solver(16, 0.1, 0.001, 3, optimizer) for optimizer in TESTS_OPTIMIZERS],
        )

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])
    tests += _generate_test(ExponentialModel, [(0.25, 25)])

    save_results(
        run_tests(tests=tests, workers_count=MAX_WORKERS, chunksize=256),
        "big_mono_test",
    )
