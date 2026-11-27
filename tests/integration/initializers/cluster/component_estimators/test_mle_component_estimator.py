"""Integration tests of maximum likelihood component estimation on real families."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import assume, given, settings
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import _components
from pysatl_mpest.initializers.cluster.component_estimators.mle_component_estimator import MLEComponentEstimator
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.partitions import partition_inputs, random_partition


def common_cluster() -> tuple[UnivariateFloatArray, UnivariateFloatArray]:
    """The sample ``[1, 2, 6, 100]`` and the cluster of its first three points."""

    X = np.array([1, 2, 6, 100], dtype=np.float64)
    h = np.array([1, 1, 1, 0], dtype=np.float64)
    return X, h


def log_likelihood(component: ParametricFamilyDistribution, points: UnivariateFloatArray) -> float:
    """Log-likelihood of the cluster points under a component."""

    return float(np.sum(_components.lpdf(component, points)))


@pytest.mark.parametrize(
    "make_family",
    [
        pytest.param(lambda: family(FamilyName.NORMAL), id="normal"),
        pytest.param(lambda: family(FamilyName.EXPONENTIAL), id="exponential"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM), id="uniform"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(mu=0), id="normal-fixed-mu"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(sigma=1.0), id="normal-fixed-sigma"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(lower_bound=0.0), id="uniform-fixed-lower"),
        pytest.param(lambda: family(FamilyName.EXPONENTIAL).view(), id="exponential-empty-view"),
    ],
)
def test_supports_base_parametrizations_and_their_views(make_family: Callable[[], ParametricFamily]) -> None:
    assert MLEComponentEstimator().supports(make_family()) is True


@pytest.mark.parametrize(
    "make_family",
    [
        pytest.param(lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar"), id="normal-meanVar"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(parametrization_name="meanPrec"), id="normal-meanPrec"),
        pytest.param(
            lambda: family(FamilyName.NORMAL).view(parametrization_name="exponential"), id="normal-exponential"
        ),
        pytest.param(lambda: family(FamilyName.EXPONENTIAL).view(parametrization_name="scale"), id="exponential-scale"),
        pytest.param(
            lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(parametrization_name="meanWidth"),
            id="uniform-meanWidth",
        ),
        pytest.param(
            lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(parametrization_name="minRange"),
            id="uniform-minRange",
        ),
    ],
)
def test_does_not_support_other_parametrizations(make_family: Callable[[], ParametricFamily]) -> None:
    assert MLEComponentEstimator().supports(make_family()) is False


@pytest.mark.parametrize(
    "make_family",
    [
        pytest.param(lambda: family(FamilyName.NORMAL), id="normal"),
        pytest.param(lambda: family(FamilyName.EXPONENTIAL), id="exponential"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM), id="uniform"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(mu=0), id="normal-fixed-mu"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(sigma=1.0), id="normal-fixed-sigma"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(lower_bound=0.0), id="uniform-fixed-lower"),
    ],
)
def test_returns_exactly_free_parameters_of_family(make_family: Callable[[], ParametricFamily]) -> None:
    X, h = common_cluster()
    tested_family = make_family()

    free = MLEComponentEstimator().estimate(X, h, tested_family)

    assert set(free) == set(_components.family_free_parameter_names(tested_family))
    assert all(type(value) is float for value in free.values())


@pytest.mark.parametrize(
    ("make_family", "x_values", "h_values", "expected"),
    [
        pytest.param(
            lambda: family(FamilyName.NORMAL).view(mu=0),
            [4, 5, 6, 100],
            [1, 1, 1, 0],
            {"sigma": 5.066228051190222},
            id="normal-fixed-mu",
        ),
        pytest.param(
            lambda: family(FamilyName.NORMAL),
            [4, 5, 6, 100],
            [1, 1, 1, 0],
            {"mu": 5.0, "sigma": 0.816496580927726},
            id="normal",
        ),
        pytest.param(
            lambda: family(FamilyName.EXPONENTIAL),
            [1, 2, 6, 100],
            [1, 1, 1, 0],
            {"lambda_": 1 / 3},
            id="exponential",
        ),
        pytest.param(
            lambda: family(FamilyName.CONTINUOUS_UNIFORM),
            [2, 7, 3, 100, -50],
            [1, 1, 1, 0, 0],
            {"lower_bound": 2.0, "upper_bound": 7.0},
            id="uniform",
        ),
    ],
)
def test_estimates_literal_values(
    make_family: Callable[[], ParametricFamily],
    x_values: list[float],
    h_values: list[float],
    expected: dict[str, float],
) -> None:
    X = np.array(x_values, dtype=np.float64)
    h = np.array(h_values, dtype=np.float64)

    free = MLEComponentEstimator().estimate(X, h, make_family())

    assert free == expected


def test_unsupported_parametrization_raises_dispatch_error() -> None:
    X, h = common_cluster()
    mean_var = family(FamilyName.NORMAL).view(parametrization_name="meanVar")

    with pytest.raises(NotImplementedError, match=r"'Normal'.*'meanVar'"):
        MLEComponentEstimator().estimate(X, h, mean_var)


@pytest.mark.parametrize(
    "make_family",
    [
        pytest.param(lambda: family(FamilyName.NORMAL), id="normal"),
        pytest.param(lambda: family(FamilyName.EXPONENTIAL), id="exponential"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM), id="uniform"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(mu=0), id="normal-fixed-mu"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(sigma=1.0), id="normal-fixed-sigma"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(lower_bound=0.0), id="uniform-fixed-lower"),
    ],
)
def test_result_builds_component_of_same_family_with_fixed_parameters(
    make_family: Callable[[], ParametricFamily],
) -> None:
    X, h = common_cluster()
    tested_family = make_family()
    estimator = MLEComponentEstimator()

    component = tested_family.distribution(**estimator.estimate(X, h, tested_family))

    assert _components.component_family_key(component) == _components.family_key(tested_family)
    assert _components.fixed_parameters(component) == _components.family_fixed_parameters(tested_family)


@settings(deadline=None)
@given(partition_inputs())
def test_normal_estimate_is_cluster_likelihood_maximum(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs
    normal = family(FamilyName.NORMAL)
    h = random_partition(len(X), k, seed)[:, 0]
    points = X[h > 0]

    free = MLEComponentEstimator().estimate(X, h, normal)
    assume(free["sigma"] > 1e-3)  # noqa: PLR2004
    best = log_likelihood(normal.distribution(**free), points)

    for name in ("mu", "sigma"):
        for factor in (0.99, 1.01):
            shifted = {**free, name: free[name] * factor}
            assert best >= log_likelihood(normal.distribution(**shifted), points) - 1e-9 * abs(best)


@settings(deadline=None)
@given(partition_inputs())
def test_exponential_estimate_is_cluster_likelihood_maximum(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs
    X = np.abs(X) + 1.0
    expon = family(FamilyName.EXPONENTIAL)
    h = random_partition(len(X), k, seed)[:, 0]
    points = X[h > 0]

    free = MLEComponentEstimator().estimate(X, h, expon)
    best = log_likelihood(expon.distribution(**free), points)

    for factor in (0.99, 1.01):
        shifted = expon.distribution(lambda_=free["lambda_"] * factor)
        assert best >= log_likelihood(shifted, points) - 1e-9 * abs(best)


@settings(deadline=None)
@given(partition_inputs())
def test_uniform_estimate_is_cluster_likelihood_maximum(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)
    h = random_partition(len(X), k, seed)[:, 0]
    points = X[h > 0]

    free = MLEComponentEstimator().estimate(X, h, uniform)
    lower, upper = free["lower_bound"], free["upper_bound"]
    assume(upper - lower > 1e-3)  # noqa: PLR2004

    best = log_likelihood(uniform.distribution(**free), points)
    delta = 0.01 * (upper - lower)

    assert np.isfinite(best)
    assert log_likelihood(uniform.distribution(lower_bound=lower + delta, upper_bound=upper), points) == -np.inf
    assert log_likelihood(uniform.distribution(lower_bound=lower, upper_bound=upper - delta), points) == -np.inf
    assert log_likelihood(uniform.distribution(lower_bound=lower - delta, upper_bound=upper + delta), points) < best
