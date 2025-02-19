"""Module which contains experiment of moving one distribution against other"""

import random

import numpy as np

from examples.config import MAX_WORKERS, TESTS_OPTIMIZERS
from examples.mono_test_generator import Clicker
from examples.utils import Test, init_solver, run_tests, save_results
from mpest import Distribution, MixtureDistribution, Problem
from mpest.models import GaussianModel, WeibullModelExp

# Gaussian

gaussian_start_params = [(0.0, 3.0), (-10.0, 3.0), (10.0, 3.0)]
weibull_start_params = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (5.0, 1.0)]
sizes = [50, 100, 200, 500, 1000]
BASE_SIZE = 2048
TESTS_PER_COND = 8
TESTS_PER_SIZE = 8

clicker = Clicker()
gaussian = GaussianModel()
weibull = WeibullModelExp()

tests = []

for sp in gaussian_start_params:
    main_distr = list(gaussian.generate(np.array(sp), BASE_SIZE // 2, normalized=False))
    for second_sp in np.linspace(sp[0] - 5, sp[0] + 5, num=8, endpoint=True):
        x = main_distr + list(gaussian.generate(np.array((second_sp, 3.0)), BASE_SIZE // 2, normalized=False))
        random.shuffle(x)

        start_params_borders = [
            (min(sp[0], second_sp) - 1.0, max(sp[0], second_sp) + 1.0),
            (1.0, 5.0),
        ]

        for size in sizes:
            for _ in range(TESTS_PER_COND):
                samples = random.sample(x, size)
                for _ in range(TESTS_PER_SIZE):
                    start_params = [
                        np.array([random.uniform(border[0], border[1]) for border in start_params_borders])
                        for _ in range(2)
                    ]
                    tests.append(
                        Test(
                            clicker.click(),
                            np.array(x),
                            MixtureDistribution.from_distributions(
                                [
                                    Distribution(
                                        gaussian,
                                        np.array(sp),
                                    ),
                                    Distribution(
                                        gaussian,
                                        np.array((second_sp, 3.0)),
                                    ),
                                ]
                            ),
                            Problem(
                                np.array(samples),
                                MixtureDistribution.from_distributions(
                                    [
                                        Distribution(
                                            gaussian,
                                            params,
                                        )
                                        for params in start_params
                                    ]
                                ),
                            ),
                            [init_solver(16, 0.1, 0.001, 3, optimizer) for optimizer in TESTS_OPTIMIZERS],
                            1,
                        )
                    )

for sp in weibull_start_params:
    main_distr = list(weibull.generate(np.array(sp), BASE_SIZE // 2, normalized=False))
    for second_sp in np.linspace(max(sp[0] - 5, 0.1), sp[0] + 5, num=8, endpoint=True):
        x = main_distr + list(weibull.generate(np.array((second_sp, 1.0)), BASE_SIZE // 2, normalized=False))
        random.shuffle(x)

        start_params_borders = [
            (max(min(sp[0], second_sp) - 1.0, 0.1), max(sp[0], second_sp) + 1.0),
            (0.5, 2.0),
        ]

        for size in sizes:
            for _ in range(TESTS_PER_COND):
                samples = random.sample(x, size)
                for _ in range(TESTS_PER_SIZE):
                    start_params = [
                        np.array([random.uniform(border[0], border[1]) for border in start_params_borders])
                        for _ in range(2)
                    ]
                    tests.append(
                        Test(
                            clicker.click(),
                            np.array(x),
                            MixtureDistribution.from_distributions(
                                [
                                    Distribution(
                                        weibull,
                                        np.array(sp),
                                    ),
                                    Distribution(
                                        weibull,
                                        np.array((second_sp, 3.0)),
                                    ),
                                ]
                            ),
                            Problem(
                                np.array(samples),
                                MixtureDistribution.from_distributions(
                                    [
                                        Distribution(
                                            weibull,
                                            params,
                                        )
                                        for params in start_params
                                    ]
                                ),
                            ),
                            [init_solver(16, 0.1, 0.001, 3, optimizer) for optimizer in TESTS_OPTIMIZERS],
                            1,
                        )
                    )

save_results(
    run_tests(tests=tests, workers_count=MAX_WORKERS, chunksize=64),
    "diff_test",
)
