"""
Unit test module which tests mixture of two distributions parameter estimation
with equally probable prior probabilities
"""

# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import numpy as np
import pytest

from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)
from mpest.utils import Factory
from tests.tests_l_moments.l_moments_utils import run_test
from tests.utils import check_for_params_error_tolerance


def idfunc(vals):
    """Function for customizing pytest ids"""

    if isinstance(vals, Factory):
        return vals.cls().name
    if isinstance(vals, list):
        return vals
    return f"{vals}"


@pytest.mark.parametrize(
    "model_factory, params, start_params, size, deviation, expected_error",
    [
        (
            Factory(WeibullModelExp),
            [(0.5, 1.0), (1.0, 0.5)],
            [(1.0, 1.0), (0.5, 1.5)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(WeibullModelExp),
            [(0.5, 0.5), (2.0, 1.0)],
            [(0.1, 1.0), (1.0, 2.0)],
            500,
            0.01,
            0.2,
        ),
        (
            Factory(GaussianModel),
            [(0.0, 5.0), (1.0, 1.0)],
            [(1.0, 5.0), (-1.0, 5.0)],
            500,
            0.01,
            0.33,
        ),
        (
            Factory(GaussianModel),
            [(4.0, 5.0), (3.0, 2.0)],
            [(3.0, 5.0), (3.5, 3.0)],
            500,
            0.01,
            0.41,
        ),
        (
            Factory(ExponentialModel),
            [(1.0,), (2.0,)],
            [(0.5,), (1.5,)],
            500,
            0.01,
            0.1,
        ),
        (
            Factory(ExponentialModel),
            [(2.0,), (5.0,)],
            [(3.0,), (1.0,)],
            500,
            0.01,
            0.25,
        ),
    ],
    ids=idfunc,
)
def test_two_same_distributions_simple(
    model_factory: Factory[AModelWithGenerator],
    params,
    start_params,
    size: int,
    deviation: float,
    expected_error: float,
):
    """Runs mixture of two distributions parameter estimation unit test"""

    np.random.seed(42)

    models = [model_factory.construct() for _ in range(len(params))]

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    x = []
    for model, param in zip(models, params):
        x += list(model.generate(param, size, normalized=0))
    np.random.shuffle(x)
    x = np.array(x)

    base_mixture = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, params)]
    )

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, param) for model, param in zip(models, start_params)]
        ),
    )

    result = run_test(problem=problem, deviation=deviation)
    assert check_for_params_error_tolerance([result], base_mixture, expected_error)
