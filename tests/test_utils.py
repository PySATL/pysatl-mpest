import random
import time
from typing import NamedTuple
import pickle
import sys
import numpy as np
from tqdm.contrib.concurrent import process_map

# fmt: off

sys.path.insert(1, "../src")

from em import EM
from distribution import Distribution
import utils
from models import WeibullModelExp, GaussianModel, Model
from optimizer import Optimizer, ScipyNewtonCG

# fmt: on


class Test(NamedTuple):
    number: int

    start_distributions: list[Distribution]
    base_distributions: list[Distribution]

    base_data: utils.Samples
    data: utils.Samples

    k: int

    runs_per_test: int

    deviation: float
    max_step: int
    prior_probability_threshold: float
    prior_probability_threshold_step: int
    optimizer: type[Optimizer]


class TestResult(NamedTuple):
    test: Test
    result: EM.Result
    time: float


class Clicker:
    def __init__(self):
        self._counter = -1

    def get(self):
        self._counter += 1
        return self._counter


def run_test(test: Test) -> TestResult:
    times = []

    for _ in range(test.runs_per_test):
        start = time.perf_counter()
        result = EM.em_algo(
            test.data,
            test.start_distributions,
            test.k,
            deviation=test.deviation,
            max_step=test.max_step,
            prior_probability_threshold=test.prior_probability_threshold,
            prior_probability_threshold_step=test.prior_probability_threshold_step,
            optimizer=test.optimizer,
        )

        stop = time.perf_counter()
        times.append(stop - start)

    return TestResult(test, result, float(np.mean(times)))


def run_tests(
    tests: list[Test], workers_count: int, shuffled: bool = True
) -> list[TestResult]:
    if not shuffled:
        return process_map(run_test, tests, max_workers=workers_count)

    _tests = list(tests)
    random.shuffle(_tests)
    results: list[TestResult] = process_map(run_test, _tests, max_workers=workers_count)
    results.sort(key=lambda t: t.test.number)
    return results


def generate_mono_test(
    model: type[Model],
    o_borders_for_data: list[tuple[float, float]],
    clicker: Clicker,
    o_borders_for_start_params: list[tuple[float, float]] | None = None,
    k_list: list[int] = [1, 2, 3, 4, 5],
    sizes: list[int] = [50, 100, 200, 500],
    distribution_count: int = 1,
    base_size: int = 1024,
    tests_per_cond: int = 1,
    runs_per_test: int = 3,
    deviation: float = 0.01,
    max_step: int = 16,
    prior_probability_threshold: float = 0.001,
    prior_probability_threshold_step: int = 3,
    optimizer: type[Optimizer] = ScipyNewtonCG,
) -> list[Test]:
    new_tests: list[Test] = []

    for k in k_list:
        for _ in range(distribution_count):
            per_model = base_size // k
            base_params = []
            x = []
            for _ in range(k):
                o = np.array(
                    [
                        random.uniform(border[0], border[1])
                        for border in o_borders_for_data
                    ]
                )
                x += list(model.generate(o, per_model))
                base_params.append(o)

            random.shuffle(x)
            base = np.array(x)

            if o_borders_for_start_params is not None:
                params_borders = o_borders_for_start_params
            else:
                params_borders = o_borders_for_data

            for size in sizes:
                for _ in range(tests_per_cond):
                    start_params = [
                        np.array(
                            [
                                random.uniform(border[0], border[1])
                                for border in params_borders
                            ]
                        )
                        for _ in range(k)
                    ]
                    new_tests.append(
                        Test(
                            clicker.get(),
                            [
                                Distribution(
                                    model, model.params_convert_to_model(params)
                                )
                                for params in start_params
                            ],
                            [
                                Distribution(
                                    model, model.params_convert_to_model(params)
                                )
                                for params in base_params
                            ],
                            base,
                            np.array(random.sample(x, size)),
                            k,
                            runs_per_test,
                            deviation,
                            max_step,
                            prior_probability_threshold,
                            prior_probability_threshold_step,
                            optimizer,
                        )
                    )
    return new_tests


def save_results(results: list[TestResult], name: str) -> None:
    with open(f"results/{name}.pkl", "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def open_results(name: str) -> list[TestResult]:
    with open(f"results/{name}.pkl", "rb") as f:
        return pickle.load(f)
