"""Unit test module which tests mixture of one distribution parameter estimation"""

# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments

import numpy as np
import pytest

from mpest.distribution import Distribution
from mpest.mixture_distribution import MixtureDistribution
from mpest.models import (
    AModelWithGenerator,
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)
from mpest.problem import Problem
from tests.tests_l_moments.l_moments_utils import run_test
from tests.utils import check_for_params_error_tolerance


def idfunc(vals):
    """Function for customizing pytest ids"""

    if issubclass(type(vals), AModelWithGenerator):
        return vals.name
    if isinstance(vals, tuple):
        return vals
    return f"{vals}"


@pytest.mark.parametrize(
    "model, params, start_params, size, deviation, expected_error",
    [
        (WeibullModelExp(), (0.5, 0.5), (0.3, 2.0), 500, 0.01, 0.051),
        (WeibullModelExp(), (1.0, 1.0), (0.5, 2.0), 500, 0.01, 0.05),
        (GaussianModel(), (0.0, 5.0), (-1.0, 4.0), 500, 0.01, 0.16),
        (GaussianModel(), (1.0, 5.0), (2.0, 1.0), 500, 0.01, 0.15),
        (ExponentialModel(), (1.0,), (0.5,), 500, 0.01, 0.05),
        (ExponentialModel(), (2.0,), (3.0,), 500, 0.01, 0.05),
    ],
    ids=idfunc,
)
def test_one_distribution(
    model: AModelWithGenerator,
    params,
    start_params,
    size: int,
    deviation: float,
    expected_error: float,
):
    """Runs mixture of one distribution parameter estimation unit test"""

    np.random.seed(42)

    params = np.array(params)
    start_params = np.array(start_params)

    base_model = Distribution(model, params)
    x = base_model.generate(size)

    problem = Problem(
        samples=x,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, start_params)]
        ),
    )

    result = run_test(problem=problem, deviation=deviation)
    assert check_for_params_error_tolerance(
        [result], MixtureDistribution.from_distributions([base_model]), expected_error
    )
