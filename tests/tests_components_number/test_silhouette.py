"""Unit test module which test the Silhouette method"""

import pytest

from mpest.components_number.methods import Silhouette
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from tests.tests_components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [WeibullModelExp(), GaussianModel()],
            [[2.0, 10.0], [5.0, 1.0]],
            [0.6, 0.4],
            200,
            10,
        ),
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[5.0, 3.0], [2.0, 6.0], [1.0, 0.5]],
            [0.3, 0.3, 0.3],
            500,
            10,
        ),
        (
            [ExponentialModel(), GaussianModel(), GaussianModel(), GaussianModel()],
            [[0.5], [1.0, 0.5], [3.0, 10.0], [5.0, 1.0]],
            [0.5, 0.3, 0.1, 0.1],
            500,
            10,
        ),
    ],
)
def test_correct_estimating(models, params, prior, size, kmax):
    """Runs the Silhouette method with a positive outcome"""
    assert run_test(
        models, params, prior, size, Silhouette(kmax, random_state=42)
    ) == len(models)


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [WeibullModelExp(), WeibullModelExp(), ExponentialModel()],
            [[1.0, 2.0], [5.0, 1.0], [1.0]],
            [0.33, 0.33, 0.33],
            200,
            10,
        ),
        ([ExponentialModel()], [[0.5]], [1.0], 500, 10),
        (
            [ExponentialModel(), ExponentialModel(), WeibullModelExp()],
            [[0.5], [3.0], [3.0, 0.5]],
            [0.4, 0.5, 0.1],
            1000,
            10,
        ),
    ],
)
def test_incorrect_estimating(models, params, prior, size, kmax):
    """Runs the Silhouette method with a negative outcome"""
    assert run_test(
        models, params, prior, size, Silhouette(kmax, random_state=42)
    ) != len(models)
