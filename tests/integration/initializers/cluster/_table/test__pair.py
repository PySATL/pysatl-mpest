"""Integration tests of pair evaluation with the MLE component estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import _components
from pysatl_mpest.initializers.cluster._table._pair import PairEvaluation, evaluate_pair
from pysatl_mpest.initializers.cluster.component_estimators.mle_component_estimator import MLEComponentEstimator
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.partitions import partition_inputs, random_partition


@pytest.fixture
def normal() -> ParametricFamily:
    """The Normal family without a view."""

    return family(FamilyName.NORMAL)


@pytest.fixture
def expon() -> ParametricFamily:
    """The Exponential family without a view."""

    return family(FamilyName.EXPONENTIAL)


@pytest.fixture
def uniform() -> ParametricFamily:
    """The continuous uniform family without a view."""

    return family(FamilyName.CONTINUOUS_UNIFORM)


def univariate(values: list[float]) -> UnivariateFloatArray:
    """One-dimensional ``float64`` array from literal values."""

    return np.asarray(values, dtype=np.float64)


def evaluate_with_mle(
    X: UnivariateFloatArray,
    h: UnivariateFloatArray,
    tested_family: ParametricFamily,
    scale: float | None = None,
) -> PairEvaluation:
    """Evaluate a pair by the real estimator, computing the scale from the sample unless it is given."""

    if scale is None:
        scale = float(np.std(X))
    return evaluate_pair(X, h, tested_family, MLEComponentEstimator(), scale)


def assert_estimated(result: PairEvaluation) -> None:
    """Assert the state of an estimated pair."""

    assert result.component is not None
    assert np.isfinite(result.loglikelihood)
    assert result.reason is None


def not_estimated_reason(result: PairEvaluation) -> str:
    """Assert the state of a pair that was not estimated and return its reason."""

    assert result.component is None
    assert result.loglikelihood == -np.inf
    assert isinstance(result.reason, str)
    return result.reason


def normal_points() -> UnivariateFloatArray:
    """Two hundred points from the normal distribution of the first case."""

    rng = np.random.default_rng(42)
    return np.asarray(rng.normal(3.0, 2.0, 200), dtype=np.float64)


def exponential_points() -> UnivariateFloatArray:
    """Two hundred points from the exponential distribution of the first case."""

    rng = np.random.default_rng(42)
    return np.asarray(rng.exponential(2.0, 200), dtype=np.float64)


def uniform_points() -> UnivariateFloatArray:
    """Two hundred points from the uniform distribution of the first case."""

    rng = np.random.default_rng(42)
    return np.asarray(rng.uniform(1.0, 5.0, 200), dtype=np.float64)


class TestEstimated:
    """Pairs for which the real estimator and the checks produce a component."""

    @pytest.mark.parametrize(
        ("make_family", "draw_points", "expected"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                normal_points,
                {"mu": (3.0, 0.3), "sigma": (2.0, 0.3)},
                id="normal",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                exponential_points,
                {"lambda_": (0.5, 0.05)},
                id="exponential",
            ),
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM),
                uniform_points,
                {"lower_bound": (1.0, 0.05), "upper_bound": (5.0, 0.05)},
                id="uniform",
            ),
        ],
    )
    def test_parameters_are_close_to_true_values(
        self,
        make_family: Callable[[], ParametricFamily],
        draw_points: Callable[[], UnivariateFloatArray],
        expected: dict[str, tuple[float, float]],
    ) -> None:
        points = draw_points()
        X = np.concatenate([points, np.array([-50.0, 100.0], dtype=np.float64)])
        h = np.concatenate([np.ones(len(points), dtype=np.float64), np.zeros(2, dtype=np.float64)])

        result = evaluate_with_mle(X, h, make_family())

        assert_estimated(result)
        assert result.component is not None
        free = _components.free_parameters(result.component)
        for name, (value, tolerance) in expected.items():
            assert free[name] == pytest.approx(value, abs=tolerance)
        assert result.loglikelihood == pytest.approx(
            float(np.sum(_components.lpdf(result.component, points))), rel=1e-12
        )

    def test_fixed_mean_is_used_for_spread(self, normal: ParametricFamily) -> None:
        X = univariate([4.0, 5.0, 6.0, 100.0])
        h = univariate([1.0, 1.0, 1.0, 0.0])

        result = evaluate_with_mle(X, h, normal.view(mu=0))

        assert_estimated(result)
        assert result.component is not None
        assert _components.all_parameters(result.component) == {
            "mu": 0,
            "sigma": pytest.approx(5.066228051190222, rel=1e-12),
        }
        assert result.loglikelihood == pytest.approx(-9.124605299392378, rel=1e-12)

    def test_one_distinct_value_is_enough_for_one_free_parameter(self, expon: ParametricFamily) -> None:
        X = univariate([0.1] * 7 + [5.0, 6.0])
        h = univariate([1.0] * 7 + [0.0, 0.0])

        result = evaluate_with_mle(X, h, expon)

        assert_estimated(result)
        assert result.component is not None
        assert _components.free_parameters(result.component)["lambda_"] == pytest.approx(10.0, rel=1e-12)

    # Temporary until the M-step tolerance becomes relative (issue #51).
    def test_m_step_tolerance_does_not_apply_to_uniform(self, uniform: ParametricFamily) -> None:
        X = 1e-10 * np.array([0.0, 1, 2, 10, 11, 12], dtype=np.float64)
        h = univariate([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        result = evaluate_with_mle(X, h, uniform)

        assert_estimated(result)


class TestDoesNotCover:
    """Pairs whose component does not cover every point of the cluster."""

    @pytest.mark.parametrize(
        ("make_family", "x_values", "parameter_assertion", "example"),
        [
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                [-1.0, 5.0, 20.0],
                lambda component: _components.free_parameters(component)["lambda_"] == pytest.approx(0.5),
                "for example -1.0.",
                id="exponential",
            ),
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(lower_bound=3.0),
                [1.0, 5.0, 20.0],
                lambda component: _components.all_parameters(component) == {"lower_bound": 3.0, "upper_bound": 5.0},
                "for example 1.0.",
                id="uniform-fixed-lower",
            ),
        ],
    )
    def test_point_outside_support_keeps_component_and_names_it(
        self,
        make_family: Callable[[], ParametricFamily],
        x_values: list[float],
        parameter_assertion: Callable[[ParametricFamilyDistribution], bool],
        example: str,
    ) -> None:
        X = univariate(x_values)
        h = univariate([1.0, 1.0, 0.0])

        result = evaluate_with_mle(X, h, make_family())

        assert result.component is not None
        assert parameter_assertion(result.component)
        assert result.loglikelihood == -np.inf
        assert result.reason is not None
        assert result.reason.endswith(example)


class TestNotEstimated:
    """Pairs rejected before a component or a finite log-likelihood can be produced."""

    def test_constraint_violation_is_rejected(self, expon: ParametricFamily) -> None:
        X = univariate([-5.0, 1.0, 20.0])
        h = univariate([1.0, 1.0, 0.0])

        result = evaluate_with_mle(X, h, expon)

        reason = not_estimated_reason(result)
        assert reason.startswith(
            "the estimated parameters {'lambda_': -0.5} violate the constraints of the parametrization: "
        )

    @pytest.mark.parametrize(
        "make_family",
        [
            pytest.param(lambda: family(FamilyName.NORMAL), id="normal"),
            pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM), id="uniform"),
        ],
    )
    def test_fewer_distinct_values_than_free_parameters(self, make_family: Callable[[], ParametricFamily]) -> None:
        X = univariate([0.1] * 7 + [5.0, 6.0])
        h = univariate([1.0] * 7 + [0.0, 0.0])

        result = evaluate_with_mle(X, h, make_family())

        assert not_estimated_reason(result) == (
            "the cluster has 1 distinct values, fewer than the 2 free parameters of the family."
        )

    @pytest.mark.parametrize(
        ("make_family", "x_values", "h_values", "scale"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                [0.1] * 6 + [float(np.nextafter(0.1, 1)), 5.0, 6.0, 7.0],
                [1.0] * 7 + [0.0, 0.0, 0.0],
                None,
                id="normal",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                [1e-8, 3e-8],
                [1.0, 1.0],
                1.0,
                id="exponential",
            ),
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM),
                [5.0, float(np.nextafter(5.0, 6.0))],
                [1.0, 1.0],
                1.0,
                id="uniform",
            ),
        ],
    )
    def test_too_narrow_component_is_rejected(
        self,
        make_family: Callable[[], ParametricFamily],
        x_values: list[float],
        h_values: list[float],
        scale: float | None,
    ) -> None:
        X = univariate(x_values)
        h = univariate(h_values)

        result = evaluate_with_mle(X, h, make_family(), scale)

        assert not_estimated_reason(result).startswith("the interquartile width of the component ")

    # Temporary until the M-step tolerance becomes relative (issue #51).
    @pytest.mark.parametrize(
        ("make_family", "factor", "quantity"),
        [
            pytest.param(lambda: family(FamilyName.NORMAL), 1e-5, "variance", id="normal"),
            pytest.param(lambda: family(FamilyName.EXPONENTIAL), 1e-10, "mean", id="exponential"),
        ],
    )
    def test_parameter_below_m_step_tolerance_is_rejected(
        self,
        make_family: Callable[[], ParametricFamily],
        factor: float,
        quantity: str,
    ) -> None:
        X = factor * np.array([0.0, 1, 2, 10, 11, 12], dtype=np.float64)
        h = univariate([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        result = evaluate_with_mle(X, h, make_family())

        reason = not_estimated_reason(result)
        assert reason.startswith(f"the {quantity} of the component ")
        assert reason.endswith("rescale the data.")

    def test_zero_cluster_mean_gives_non_finite_estimate_without_warnings(self, expon: ParametricFamily) -> None:
        X = univariate([-1.0, 1.0, 5.0])
        h = univariate([1.0, 1.0, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = evaluate_with_mle(X, h, expon)

        assert not_estimated_reason(result) == "the estimated parameters are not finite: {'lambda_': inf}."


@pytest.mark.parametrize(
    "make_family",
    [
        pytest.param(lambda: family(FamilyName.NORMAL), id="normal"),
        pytest.param(lambda: family(FamilyName.EXPONENTIAL), id="exponential"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM), id="uniform"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(mu=0.0), id="normal-fixed-mu"),
        pytest.param(lambda: family(FamilyName.NORMAL).view(sigma=2.0), id="normal-fixed-sigma"),
        pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(lower_bound=0.0), id="uniform-fixed-lower"),
    ],
)
@settings(deadline=None)
@given(inputs=partition_inputs())
def test_always_returns_one_of_the_three_pair_states(
    make_family: Callable[[], ParametricFamily], inputs: tuple[UnivariateFloatArray, int, int]
) -> None:
    X, k, seed = inputs
    H = random_partition(len(X), k, seed)
    tested_family = make_family()

    for c in range(k):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = evaluate_pair(X, H[:, c], tested_family, MLEComponentEstimator(), float(np.std(X)))

        if result.component is None:
            assert result.loglikelihood == -np.inf
            assert isinstance(result.reason, str)
        else:
            assert _components.component_family_key(result.component) == _components.family_key(tested_family)
            assert (result.reason is None) == bool(np.isfinite(result.loglikelihood))
