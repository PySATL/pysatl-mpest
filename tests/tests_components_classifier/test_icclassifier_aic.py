"""Unit test module which test IC-based Classifier with AIC"""

import pytest

from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.optimizers.scipy_slsqp import ScipySLSQP
from mpest.preprocessing.criterions import AIC
from mpest.preprocessing.components_classifier import ICClassifier
from tests.tests_components_classifier.components_classifier_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [ExponentialModel(), ExponentialModel(), ExponentialModel()],
            [[3.0], [1.5], [1.0]],
            [0.33, 0.33, 0.33],
            200,
            [0, 0, 3]
        ),
        (
            [GaussianModel(), GaussianModel(), GaussianModel(), GaussianModel()],
            [[0.0, 0.5], [2.0, 1.0], [4.0, 1.5], [7.0, 5.0]],
            [0.25, 0.2, 0.15, 0.35],
            500,
            [4, 0, 0]
        ),
        (
            [ExponentialModel(), GaussianModel()],
            [[0.5], [2.0, 0.5]],
            [0.5, 0.5],
            1000,
            [1, 0, 1]
        )
    ]
)
def test_correct_prediction(models, params, prior, size, counter):
    """Runs IC-based Classifier (AIC) with a positive outcome"""
    assert all(run_test(
        models,
        params,
        prior,
        size,
        ICClassifier(AIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()))) == counter)


@pytest.mark.parametrize(
    "models, params, prior, size, counter",
    [
        (
            [WeibullModelExp(), GaussianModel()],
            [[2.0, 0.5], [2.0, 1.0]],
            [0.6, 0.4],
            200,
            [1, 1, 0]
        ),
        (
            [ExponentialModel()],
            [[7.5]],
            [1.0],
            500,
            [0, 0, 1]
        ),
        (
            [WeibullModelExp(), WeibullModelExp(), WeibullModelExp()],
            [[3.0, 1.0], [1.5, 0.5], [0.5, 2.0]],
            [0.33, 0.33, 0.33],
            1000,
            [0, 3, 0]
        )
    ]
)
def test_incorrect_prediction(models, params, prior, size, counter):
    """Runs IC-based Classifier (AIC) with a negative outcome"""
    assert not all(run_test(
        models,
        params,
        prior,
        size,
        ICClassifier(AIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()))) == counter)