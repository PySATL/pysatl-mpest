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

from tests.helpers.estimator_data import exponential, normal
from tests.helpers.families import family


def uniform(lower_bound: float, upper_bound: float) -> ParametricFamilyDistribution:
    """Create a continuous uniform component on ``[lower_bound, upper_bound]``."""

    return family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.mark.parametrize(
    ("make_components", "X", "expected_uncovered"),
    [
        pytest.param(lambda: [exponential(1.0), exponential(0.5)], [-3.4, 0.0, 2.0, -1.0], [0, 3], id="exponential"),
        pytest.param(
            lambda: [uniform(0.0, 1.0), uniform(2.0, 3.0)],
            [-0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5],
            [0, 4, 7],
            id="uniform",
        ),
        pytest.param(lambda: [normal(0.0, 1.0), normal(5.0, 1.0)], [-30.0, 0.0, 5.0, 40.0], [], id="normal"),
    ],
)
def test_reports_points_outside_support_of_every_component(
    make_components: Callable[[], list[ParametricFamilyDistribution]], X: list[float], expected_uncovered: list[int]
) -> None:
    coverage = sample_coverage(MixtureModel(make_components()), np.array(X))

    np.testing.assert_array_equal(coverage.uncovered, expected_uncovered)
    np.testing.assert_array_equal(coverage.undefined, [])


def test_does_not_report_point_covered_by_a_neighboring_component() -> None:
    mixture = MixtureModel([normal(0.0, 1.0), exponential(1.0)])
    X = np.array([-3.4, 0.0, 2.0])

    coverage = sample_coverage(mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, [])
    np.testing.assert_array_equal(coverage.undefined, [])


def test_agrees_with_sample_loglikelihood() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-3, 1, 300), rng.exponential(1.0, 300)])

    exponential_mixture = MixtureModel([exponential(1.0), exponential(0.5)])
    coverage = sample_coverage(exponential_mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, np.flatnonzero(X < 0))
    np.testing.assert_array_equal(coverage.undefined, [])
    assert exponential_mixture.loglikelihood(X) == -np.inf

    mixed_mixture = MixtureModel([normal(-3.0, 1.0), exponential(1.0)])
    coverage = sample_coverage(mixed_mixture, X)

    np.testing.assert_array_equal(coverage.uncovered, [])
    np.testing.assert_array_equal(coverage.undefined, [])
    assert np.isfinite(mixed_mixture.loglikelihood(X))
