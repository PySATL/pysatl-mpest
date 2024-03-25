import random
import sys
import numpy as np

from test_utils import Test, Clicker, generate_mono_test, run_tests, save_results

# fmt: off

sys.path.insert(1, "../src")

from models import ExponentialModel, GaussianModel, Model
from distribution import Distribution
from optimizer import ScipyNewtonCG

# fmt: on

if __name__ == "__main__":
    models = [
        (ExponentialModel, ((0.25, 15),)),
        (GaussianModel, ((-15, 15), (0.25, 15))),
    ]
    k_list = [(1, 1), (1, 2), (2, 2), (2, 3)]
    sizes: list[int] = [50, 100, 200]
    max_step = 32
    distribution_count = 32
    base_size = 1024
    tests_per_cond = 32
    runs_per_test = 1

    tests = []

    for xk, yk in k_list:
        for _ in range(distribution_count):
            per_model = base_size // (xk + yk)
            base_distributions = []
            x = []
            for k, model in zip((xk, yk), models):
                for _ in range(k):
                    o = np.array(
                        [random.uniform(border[0], border[1]) for border in model[1]]
                    )
                    x += list(model[0].generate(o, per_model))
                    base_distributions.append(
                        Distribution(model[0], model[0].params_convert_to_model(o))
                    )
            base = np.array(x)
            for size in sizes:
                for _ in range(tests_per_cond):
                    start_distributions = []
                    for k, model in zip((xk, yk), models):
                        for _ in range(k):
                            o = np.array(
                                [
                                    random.uniform(border[0], border[1])
                                    for border in model[1]
                                ]
                            )
                            start_distributions.append(
                                Distribution(
                                    model[0], model[0].params_convert_to_model(o)
                                )
                            )
                    tests.append(
                        Test(
                            len(tests),
                            start_distributions,
                            base_distributions,
                            base,
                            np.array(random.sample(x, size)),
                            xk + yk,
                            runs_per_test,
                            0.01,
                            max_step,
                            0.001,
                            3,
                            ScipyNewtonCG,
                        )
                    )

    result = run_tests(tests, 14)

    save_results(result, "exponential_gauss")
