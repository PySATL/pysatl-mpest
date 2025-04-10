"""Unit test module which test XMeans method with BIC"""

import pytest

from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.optimizers.scipy_slsqp import ScipySLSQP
from mpest.preprocessing.components_number.criterions import BIC
from mpest.preprocessing.components_number.methods import XMeans
from tests.tests_components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [GaussianModel(), WeibullModelExp()],
            [[4.0, 0.5], [1.5, 2.5]],
            [0.6, 0.4],
            500,
            6
        ),
        (
            [ExponentialModel(), WeibullModelExp()],
            [[1.5], [2.5, 3.0]],
            [0.3, 0.7],
            500,
            6
        ),
        (
            [GaussianModel(), GaussianModel(), GaussianModel()],
            [[3.0, 1.5], [-2.0, 2.0], [7.0, 1.0]],
            [0.33, 0.33, 0.33],
            500,
            6
        )
    ]
)
def test_correct_estimating(models, params, prior, size, kmax):
    """Runs XMeans method (BIC) with a positive outcome"""
    assert run_test(
        models,
        params,
        prior,
        size,
        XMeans(kmax, BIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()), random_state=42)) == len(models)


@pytest.mark.parametrize(
    "models, params, prior, size, kmax",
    [
        (
            [WeibullModelExp(), WeibullModelExp(), WeibullModelExp()],
            [[4.0, 2.0], [3.0, 1.5], [1.5, 0.5]],
            [0.25, 0.5, 0.25],
            500,
            6
        ),
        (
            [GaussianModel(), ExponentialModel()],
            [[3.0, 1.0], [1.5]],
            [0.25, 0.75],
            500,
            6
        ),
        (
            [GaussianModel(), GaussianModel()],
            [[3.0, 1.0], [3.5, 0.5]],
            [0.7, 0.3],
            500,
            6
        )
    ]
)
def test_incorrect_estimating(models, params, prior, size, kmax):
    """Runs XMeans method (BIC) with a negative outcome"""
    assert run_test(
        models,
        params,
        prior,
        size,
        XMeans(kmax, BIC(), BayesEStep(), LikelihoodMStep(ScipySLSQP()), random_state=42)) != len(models)
