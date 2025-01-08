"""Unit test module which test the Peak method"""

import pytest

from mpest.components_number.methods import Peaks
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from tests.tests_components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size",
    [
        ([GaussianModel()], [[5.0, 2.0]], [1.0], 200),
        (
            [WeibullModelExp(), WeibullModelExp(), WeibullModelExp()],
            [[5.0, 2.0], [7.0, 1.0], [11.0, 3.0]],
            [0.33, 0.33, 0.33],
            500,
        ),
        (
            [WeibullModelExp(), GaussianModel(), WeibullModelExp()],
            [[4.0, 2.0], [7.5, 2.5], [10.0, 4.0]],
            [0.2, 0.4, 0.2],
            1000,
        ),
    ],
)
def test_correct_estimating(models, params, prior, size):
    """Runs the Peak method with a positive outcome"""
    assert run_test(models, params, prior, size, Peaks()) == len(models)


@pytest.mark.parametrize(
    "models, params, prior, size",
    [
        (
            [WeibullModelExp(), WeibullModelExp(), ExponentialModel()],
            [[10.0, 1.0], [4.0, 6.0], [3.5]],
            [0.2, 0.4, 0.4],
            200,
        ),
        (
            [ExponentialModel(), ExponentialModel(), GaussianModel(), GaussianModel()],
            [[0.5], [3.5], [9.0, 0.5], [3.0, 6.0]],
            [0.1, 0.2, 0.4, 0.3],
            5000,
        ),
        (
            [GaussianModel(), WeibullModelExp()],
            [[3.0, 1.5], [7.0, 2.0]],
            [0.7, 0.3],
            1000,
        ),
    ],
)
def test_incorrect_estimating(models, params, prior, size):
    """Runs the Peak method with a negative outcome"""
    assert run_test(models, params, prior, size, Peaks()) != len(models)
