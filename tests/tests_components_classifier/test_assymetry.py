"""Unit test module which test the Assymetry Classifier"""

import pytest

from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_classifier import Assymetry
from tests.tests_components_classifier.components_classifier_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[2.0, 1.0], [5.0, 0.5], [-2.0, 3.0]],
            [0.4, 0.2, 0.2],
            200,
            [3, 0, 0]
        ),
        (
            [ExponentialModel(), WeibullModelExp()],
            [[0.5], [1.5, 2.0]],
            [0.5, 0.5],
            500,
            [0, 1, 1]
        ),
        (
            [ExponentialModel(), GaussianModel()],
            [[1.0], [2.0, 0.5]],
            [0.3, 0.7],
            1000,
            [1, 0, 1]
        )
    ]
)
def test_correct_prediction(models, params, prior, size, counter):
    """Runs the Assymetry method with a positive outcome"""
    assert all(run_test(models, params, prior, size, Assymetry(random_state=42)) == counter)


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [ExponentialModel(), WeibullModelExp()],
            [[2.0], [1.5, 3.0]],
            [0.5, 0.5],
            200,
            [0, 1, 1]
        ),
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[1.0, 1.0], [2.0, 0.5], [-2.0, 3.0]],
            [0.3, 0.3, 0.4],
            500,
            [3, 0, 0]
        ),
        (
            [GaussianModel(), WeibullModelExp()],
            [[0.5, 1.0], [0.9, 1.5]],
            [0.5, 0.5],
            1000,
            [1, 1, 0]
        )
    ]
)
def test_incorrect_prediction(models, params, prior, size, counter):
    """Runs the Assymetry method with a negative outcome"""
    assert not all(run_test(models, params, prior, size, Assymetry(random_state=42)) == counter)
