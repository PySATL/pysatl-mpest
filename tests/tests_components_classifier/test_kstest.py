"""Unit test module which test the KSTest Classifier"""

import pytest

from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_classifier import  KSTest
from tests.tests_components_classifier.components_classifier_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [ExponentialModel(), WeibullModelExp()],
            [[1.5], [1.0, 3.0]],
            [0.5, 0.5],
            200,
            [0, 1, 1]
        ),
        (
            [GaussianModel(), WeibullModelExp()],
            [[4.0, 1.0], [1.5, 1.0]],
            [0.8, 0.2],
            500,
            [1, 1, 0]
        ),
        (
            [WeibullModelExp(), WeibullModelExp(), WeibullModelExp()],
            [[1.0, 1.0], [2.0, 0.5], [3.5, 1.0]],
            [0.4, 0.3, 0.3],
            1000,
            [0, 3, 0]
        )
    ]
)
def test_correct_prediction(models, params, prior, size, counter):
    """Runs the KSTest method with a positive outcome"""
    assert all(run_test(models, params, prior, size, KSTest(random_state=42)) == counter)


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [ExponentialModel(), ExponentialModel(), ExponentialModel()],
            [[1.0], [2.0], [3.5]],
            [0.3, 0.3, 0.4],
            200,
            [0, 0, 3]
        ),
        (
            [ExponentialModel(), WeibullModelExp()],
            [[1.0], [2.0, 1.0]],
            [0.4, 0.6],
            500,
            [0, 1, 1]
        ),
        (
            [GaussianModel(), GaussianModel()],
            [[0.0, 0.5], [2.0, 1.0]],
            [0.5, 0.5],
            1000,
            [2, 0, 0]
        )
    ]
)
def test_incorrect_prediction(models, params, prior, size, counter):
    """Runs the KSTest method with a negative outcome"""
    assert not all(run_test(models, params, prior, size, KSTest(random_state=42)) == counter)
