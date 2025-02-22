"""Module which provides source for quick test of mixture distributions of single model"""

import random

import numpy as np

from examples.config import MAX_WORKERS
from examples.mono_test_generator import generate_mono_test
from examples.utils import Clicker, Test, init_solver, run_tests, save_results
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)
from mpest.optimizers import ALL_OPTIMIZERS


def run_test():
    """Runs the mixture distributions of single model quick test"""

    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(model: type[AModelWithGenerator], params_borders: list[tuple[float, float]]) -> list[Test]:
        test = generate_mono_test(
            model_t=model,
            clicker=counter,
            params_borders=params_borders,
            ks=[1, 2, 3, 4, 5],
            sizes=[50, 100, 200, 500],
            distributions_count=1,
            base_size=1024,
            tests_per_size=1,
            tests_per_cond=1,
            runs_per_test=1,
            solvers=[init_solver(16, 0.1, 0.001, 3, optimizer) for optimizer in ALL_OPTIMIZERS],
        )
        return test

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])
    tests += _generate_test(ExponentialModel, [(0.25, 25)])

    save_results(
        run_tests(
            tests=tests,
            workers_count=MAX_WORKERS,
            create_history=True,
            remember_time=True,
        ),
        "quick_test",
    )


if __name__ == "__main__":
    run_test()
