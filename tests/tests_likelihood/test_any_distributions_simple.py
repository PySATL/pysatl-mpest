"""Unit test module which tests mixture of several different distribution parameter estimation"""

# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments

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
from tests.tests_likelihood.likelihood_utils import run_test
from tests.utils import check_for_params_error_tolerance


def idfunc(vals):
    """Function for customizing pytest ids"""

    if isinstance(vals, list):
        if issubclass(type(vals[0]), AModelWithGenerator):
            return str([d.name for d in vals])
        return vals
    return f"{vals}"


@pytest.mark.parametrize(
    "models, params, start_params, size, deviation, expected_error",
    [
        (
            [WeibullModelExp(), GaussianModel()],
            [[0.5, 1.0], [5.0, 1.0]],
            [[1.5, 1.0], [1.0, 2.0]],
            1500,
            0.01,
            0.11,
        ),
        (
            [ExponentialModel(), GaussianModel()],
            [[0.5], [5.0, 1.0]],
            [[1.0], [0.0, 5.0]],
            1500,
            0.01,
            0.25,
        ),
        (
            [ExponentialModel(), WeibullModelExp()],
            [[0.5], [5.0, 1.0]],
            [[1.0], [3.0, 0.5]],
            1500,
            0.01,
            0.21,
        ),
        (
            [ExponentialModel(), GaussianModel(), WeibullModelExp()],
            [[1.0], [5.0, 1.0], [4.0, 1.0]],
            [[1.0], [7.0, 5.0], [5.0, 2.0]],
            1500,
            0.01,
            0.35,
        ),
    ],
    ids=idfunc,
)
def test(
    models,
    params,
    start_params,
    size,
    deviation,
    expected_error,
):
    """Runs mixture of several different distributions parameter estimation unit test"""

    np.random.seed(42)

    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    base_mixture = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, params)],
    )

    x = base_mixture.generate(size)

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, param) for model, param in zip(models, start_params)]
        ),
    )

    results = run_test(problem=problem, deviation=deviation)
    assert check_for_params_error_tolerance(results, base_mixture, expected_error)
