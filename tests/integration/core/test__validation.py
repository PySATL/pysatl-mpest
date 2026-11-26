"""
Integration tests for ``sample_coverage`` on real pysatl-core families.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core._validation import sample_coverage
from pysatl_mpest.core.mixture import MixtureModel
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.estimator_data import exponential, normal
from tests.helpers.families import family


def uniform(lower_bound: float, upper_bound: float) -> ParametricFamilyDistribution:
    """Create a continuous uniform component on ``[lower_bound, upper_bound]``."""
    return family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=lower_bound, upper_bound=upper_bound)


def exponential_support_case() -> tuple[MixtureModel, UnivariateFloatArray, list[int]]:
    """Two exponentials cover ``x >= 0``; negative points are uncovered."""

    mixture = MixtureModel([exponential(1.0), exponential(0.5)], weights=[0.5, 0.5])
    X = np.array([-3.4, 0.0, 2.0, -1.0])

    return mixture, X, [0, 3]


def uniform_support_case() -> tuple[MixtureModel, UnivariateFloatArray, list[int]]:
    """Two uniforms cover ``[0, 1]`` and ``[2, 3]``; the support boundaries are covered."""

    mixture = MixtureModel([uniform(0.0, 1.0), uniform(2.0, 3.0)], weights=[0.5, 0.5])
    X = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5])

    return mixture, X, [0, 4, 7]


def normal_support_case() -> tuple[MixtureModel, UnivariateFloatArray, list[int]]:
    """Normals have an unbounded support, so no point is uncovered."""

    mixture = MixtureModel([normal(0.0, 1.0), normal(5.0, 1.0)], weights=[0.5, 0.5])
    X = np.array([-30.0, 0.0, 5.0, 40.0])

    return mixture, X, []


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(exponential_support_case, id="exponential"),
        pytest.param(uniform_support_case, id="uniform"),
        pytest.param(normal_support_case, id="normal"),
    ],
)
def test_reports_points_outside_support_of_every_component(
    case: Callable[[], tuple[MixtureModel, UnivariateFloatArray, list[int]]],
) -> None:
    mixture, X, expected_uncovered = case()

    coverage = sample_coverage(mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, expected_uncovered)
    np.testing.assert_array_equal(coverage.undefined, [])


def test_does_not_report_point_covered_by_a_neighboring_component() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), exponential(1.0)], weights=[0.5, 0.5])
    X = np.array([-3.4, 0.0, 2.0])

    coverage = sample_coverage(mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, [])
    np.testing.assert_array_equal(coverage.undefined, [])


def test_agrees_with_sample_loglikelihood() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-3, 1, 300), rng.exponential(1.0, 300)])

    exponential_mixture = MixtureModel([exponential(1.0), exponential(0.5)], weights=[0.5, 0.5])
    coverage = sample_coverage(exponential_mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, np.flatnonzero(X < 0))
    np.testing.assert_array_equal(coverage.undefined, [])
    assert exponential_mixture.loglikelihood(X) == -np.inf

    mixed_mixture = MixtureModel([normal(-3.0, 1.0), exponential(1.0)], weights=[0.5, 0.5])
    coverage = sample_coverage(mixed_mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, [])
    np.testing.assert_array_equal(coverage.undefined, [])
    assert np.isfinite(mixed_mixture.loglikelihood(X))
