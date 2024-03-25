import random
import sys
import numpy as np

from test_utils import Test, Clicker, generate_mono_test, run_tests, save_results

# fmt: off

sys.path.insert(1, "../src")

from models import WeibullModelExp, GaussianModel, ExponentialModel, Model
from optimizer import ScipyNewtonCG

# fmt: on

MAX_WORKERS = 4

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(
        model: type[Model], o_borders: list[tuple[float, float]]
    ) -> list[Test]:
        return generate_mono_test(
            model,
            o_borders,
            counter,
            k_list=[1, 2, 3, 4, 5],
            sizes=[50, 100, 200, 500],
            distribution_count=1,
            base_size=1024,
            tests_per_cond=1,
            runs_per_test=1,
            deviation=0.01,
            max_step=16,
            prior_probability_threshold=0.001,
            prior_probability_threshold_step=3,
            optimizer=ScipyNewtonCG,
        )

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])
    tests += _generate_test(ExponentialModel, [(0.25, 25)])

    results = run_tests(tests, MAX_WORKERS, True)

    save_results(results, "quick_test")
