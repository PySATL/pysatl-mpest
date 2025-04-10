"""Unit test module which test XMeans method with AIC"""

import pytest

from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.optimizers.scipy_slsqp import ScipySLSQP
from mpest.preprocessing.components_number.criterions import AIC
from mpest.preprocessing.components_number.methods import XMeans
from tests.tests_components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [GaussianModel(), GaussianModel()],
            [[0.1, 0.5], [10, 1.0]],
            [0.5, 0.5],
            500,
            6,
        ),
        (
            [GaussianModel(), ExponentialModel()],
            [[5.0, 0.5], [1.0]],
            [0.5, 0.5],
            500,
            6
        ),
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[5.0, 0.5], [-2.0, 1.0], [1.0, 2.0]],
            [0.25, 0.25, 0.25],
            500,
            6
        )
    ],
)
def test_correct_estimating(models, params, prior, size, kmax):
    """Runs XMeans method (AIC) with a positive outcome"""
    assert run_test(
        models,
        params,
        prior,
        size,
        XMeans(kmax, AIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()), random_state=42)) == len(models)


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[8.0, 0.25], [-1.0, 1.5], [5.0, 3.0]],
            [0.4, 0.5, 0.1],
            500,
            6
        ),
        (
            [GaussianModel(), WeibullModelExp()],
            [[5.0, 0.25], [1.0, 1.5]],
            [0.2, 0.8],
            500,
            6
        ),
        (
            [WeibullModelExp(), WeibullModelExp(), WeibullModelExp()],
            [[5.0, 2.0], [2.0, 0.5], [0.5, 1.0]],
            [0.33, 0.33, 0.33],
            500,
            6
        )
    ]
)
def test_incorrect_estimating(models, params, prior, size, kmax):
    """Runs XMeans method (AIC) with a negative outcome"""
    assert run_test(
        models,
        params,
        prior,
        size,
        XMeans(kmax, AIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()), random_state=42)) != len(models)
