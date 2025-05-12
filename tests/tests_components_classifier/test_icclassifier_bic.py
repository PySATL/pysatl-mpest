"""Unit test module which test IC-based Classifier with BIC"""

import pytest

from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.optimizers.scipy_slsqp import ScipySLSQP
from mpest.preprocessing.criterions import BIC
from mpest.preprocessing.components_classifier import ICClassifier
from tests.tests_components_classifier.components_classifier_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [ExponentialModel()],
            [[1.0]],
            [1.0],
            200,
            [0, 0, 1]
        ),
        (
            [GaussianModel(), ExponentialModel()],
            [[3.5, 0.5], [0.5]],
            [0.6, 0.4],
            500,
            [1, 0, 1],
        ),
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[0.9, 1.0], [-2.0, 3.0], [3.0, 0.5]],
            [0.2, 0.2, 0.6],
            1000,
            [3, 0, 0]
        )
    ]
)
def test_correct_prediction(models, params, prior, size, counter):
    """Runs IC-based Classifier (BIC) with a positive outcome"""
    assert all(run_test(
        models,
        params,
        prior,
        size,
        ICClassifier(BIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()))) == counter)


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [WeibullModelExp()],
            [[3.0, 1.0]],
            [1.0],
            200,
            [0, 1, 0]
        ),
        (
            [GaussianModel(), WeibullModelExp()],
            [[3.5, 0.5], [0.5, 4.0]],
            [0.3, 0.7],
            500,
            [1, 1, 0],
        ),
        (
            [ExponentialModel(), ExponentialModel(), ExponentialModel()],
            [[7.5], [3.0], [4.0]],
            [0.8, 0.1, 0.1],
            1000,
            [0, 0, 3]
        )
    ]
)
def test_correct_prediction(models, params, prior, size, counter):
    """Runs IC-based Classifier (BIC) with a negative outcome"""
    assert not all(run_test(
        models,
        params,
        prior,
        size,
        ICClassifier(BIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()))) == counter)
