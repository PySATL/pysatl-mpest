import pytest

from tests.tests_components_number.components_num_utils import run_test
from mpest.components_number.methods import Elbow
from mpest.models import (
    ExponentialModel,
    GaussianModel,
    WeibullModelExp,
)


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [WeibullModelExp(), GaussianModel(), GaussianModel()],
            [[1.0, 0.5], [5.0, 1.0], [15.0, 2.0]],
            [0.33, 0.33, 0.33],
            200,
            15
        ),
        (
            [GaussianModel(), GaussianModel()],
            [[5.0, 2.0], [15.0, 2.0]],
            [0.6, 0.4],
            500,
            15
        ),
        (
            [WeibullModelExp(), GaussianModel(), ExponentialModel(), WeibullModelExp()],
            [[11.0, 2.5], [5.0, 3.0], [0.25], [18.0, 2.0]],
            [0.2, 0.2, 0.4, 0.2],
            1000,
            20
        ),
    ]
)
def test_correct_estimating(models, params, prior, size, kmax):
    assert run_test(models, params, prior, size, Elbow(kmax, random_state=42)) == len(models)


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[5.0, 2.0], [10.0, 2.0], [15.0, 2.0]],
            [6.0, 2.0, 2.0],
            200,
            20,
        ),
        (
            [GaussianModel(), WeibullModelExp()],
            [[5.0, 2.0], [7.0, 3.0]],
            [0.5, 0.5],
            500,
            15
        ),
        (
            [ExponentialModel(), WeibullModelExp(), WeibullModelExp()],
            [[0.5], [6.0, 5.0], [7.0, 5.0]],
            [0.1, 0.3, 0.6],
            1000,
            15
        )
    ]
)
def test_incorrect_estimating(models, params, prior, size, kmax):
    assert run_test(models, params, prior, size, Elbow(kmax, random_state=42)) != len(models)