import random
import sys
import numpy as np

from test_utils import Test, Clicker, generate_mono_test, run_tests, save_results

# fmt: off

sys.path.insert(1, "../src")

from models import WeibullModelExp, GaussianModel, Model

# fmt: on

MAX_WORKERS = 7

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    tests: list[Test] = []

    counter = Clicker()

    def _generate_test(
        model: type[Model], o_borders: list[tuple[float, float]]
    ) -> list[Test]:
        return generate_mono_test(
            model=model,
            o_borders_for_data=o_borders,
            clicker=counter,
            max_step=32,
            k_list=list(range(1, 6)),
            sizes=[50, 100, 200, 500],
            distribution_count=64,
            base_size=1024,
            tests_per_cond=16,
            runs_per_test=1,
        )

    tests += _generate_test(WeibullModelExp, [(0.25, 25), (0.25, 25)])
    tests += _generate_test(GaussianModel, [(-15, 15), (0.25, 25)])

    results = run_tests(tests, workers_count=MAX_WORKERS, shuffled=True)

    save_results(results, "big_weibull_gauss_test")
