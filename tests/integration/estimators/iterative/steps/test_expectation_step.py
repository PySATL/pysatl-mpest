"""
Integration tests for ``ExpectationStep`` on mixtures of real pysatl-core components.

The reference responsibilities are computed independently with ``scipy.stats`` in the log domain.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import ExpectationStep, PipelineState
from scipy import stats
from scipy.special import logsumexp

from tests.helpers.families import family

RTOL, ATOL = 1e-10, 1e-12


def run_step(mixture: MixtureModel, X: np.ndarray, is_soft: bool = True) -> np.ndarray:
    state = PipelineState(X=X, H=None, prev_mixture=None, curr_mixture=mixture, error=None)
    H = ExpectationStep(is_soft=is_soft).run(state).H
    assert H is not None
    return H


def reference_h(log_weights: np.ndarray, log_densities: np.ndarray) -> np.ndarray:
    """Posterior probabilities from the log-weights and the ``(n, k)`` matrix of component log-densities."""

    joint = log_densities + log_weights
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def test_soft_assignment_matches_reference() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[0.4, 0.6])
    X = np.linspace(-3.0, 8.0, 23)
    log_densities = np.column_stack([stats.norm(0.0, 1.0).logpdf(X), stats.norm(3.0, 2.0).logpdf(X)])

    H = run_step(mixture, X)

    assert_allclose(H, reference_h(np.log([0.4, 0.6]), log_densities), rtol=RTOL, atol=ATOL)
    assert_allclose(H.sum(axis=1), 1.0, rtol=RTOL)


def test_soft_assignment_of_view_components() -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=3.0)(sigma=2.0)
    mixture = MixtureModel([normal(0.0, 1.0), fixed_mu], weights=[0.5, 0.5])
    X = np.linspace(-3.0, 8.0, 23)

    H = run_step(mixture, X)

    plain = MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[0.5, 0.5])
    assert_allclose(H, run_step(plain, X), rtol=RTOL, atol=ATOL)


def test_far_tail_responsibilities_are_finite_and_exact() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[0.5, 0.5])
    X = np.array([-100.0, 100.0])
    log_densities = np.column_stack([stats.norm(0.0, 1.0).logpdf(X), stats.norm(3.0, 2.0).logpdf(X)])

    H = run_step(mixture, X)

    assert np.all(np.isfinite(H))
    assert_allclose(H, reference_h(np.log([0.5, 0.5]), log_densities), rtol=RTOL, atol=ATOL)
    assert_allclose(H.sum(axis=1), 1.0, rtol=RTOL)


def test_component_with_zero_density_gets_zero_responsibility() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), family(FamilyName.EXPONENTIAL)(lambda_=1.0)], weights=[0.5, 0.5])

    H = run_step(mixture, np.array([-2.0, -0.5]))

    assert_array_equal(H, [[1.0, 0.0], [1.0, 0.0]])


@pytest.mark.filterwarnings("error")
def test_point_outside_every_support_has_zero_responsibilities() -> None:
    exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.0)
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
    mixture = MixtureModel([exponential, uniform], weights=[0.5, 0.5])

    H = run_step(mixture, np.array([-1.0, 0.5]))

    assert not np.isnan(H).any()
    assert_array_equal(H[0], [0.0, 0.0])
    assert_allclose(H[1].sum(), 1.0)


def test_hard_assignment_is_one_hot_at_the_most_responsible_component() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), normal(6.0, 1.0)], weights=[0.5, 0.5])
    X = np.array([-1.0, 0.5, 5.0, 7.0])

    H = run_step(mixture, X, is_soft=False)

    assert_array_equal(H, [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert_array_equal(H.argmax(axis=1), run_step(mixture, X).argmax(axis=1))


def test_hard_assignment_point_outside_every_support_has_zero_responsibilities() -> None:
    exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.0)
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
    mixture = MixtureModel([exponential, uniform], weights=[0.5, 0.5])

    H = run_step(mixture, np.array([-1.0, 0.5]), is_soft=False)

    assert_array_equal(H[0], [0.0, 0.0])
    assert_array_equal(H[1], [0.0, 1.0])


def test_hard_assignment_all_points_outside_support() -> None:
    exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.0)
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
    mixture = MixtureModel([exponential, uniform], weights=[0.5, 0.5])

    H = run_step(mixture, np.array([-5.0, -2.0]), is_soft=False)

    assert_array_equal(H, [[0.0, 0.0], [0.0, 0.0]])


def test_weights_scale_the_responsibilities() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), normal(0.0, 1.0)], weights=[0.3, 0.7])

    H = run_step(mixture, np.array([0.0, 1.5]))

    assert_allclose(H, [[0.3, 0.7], [0.3, 0.7]], rtol=RTOL)
