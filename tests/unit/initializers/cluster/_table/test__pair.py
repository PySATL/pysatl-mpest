"""Unit tests for the evaluation of a pair of a family and a cluster."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import _components
from pysatl_mpest.core._numerics import NUMERICAL_TOLERANCE
from pysatl_mpest.initializers.cluster._table import _validation
from pysatl_mpest.initializers.cluster._table._pair import PairEvaluation, evaluate_pair
from pysatl_mpest.initializers.cluster.component_estimators.base import ComponentEstimator
from pysatl_mpest.initializers.exceptions import ComponentEstimationError, InitializationError
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family


class ScriptedEstimator(ComponentEstimator):
    """Component estimator double: returns the scripted answer or raises the scripted error and records the calls."""

    def __init__(self, answer: Any = None, error: BaseException | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[Any, Any, Any]] = []

    def supports(self, family: ParametricFamily) -> bool:
        return True

    def estimate(self, X: UnivariateFloatArray, h: UnivariateFloatArray, family: ParametricFamily) -> Any:
        self.calls.append((X, h, family))
        if self.error is not None:
            raise self.error
        return self.answer


class NumpyDivisionEstimator(ComponentEstimator):
    """Component estimator double: divides numpy scalars by zero inside ``estimate``."""

    def supports(self, family: ParametricFamily) -> bool:
        return True

    def estimate(self, X: UnivariateFloatArray, h: UnivariateFloatArray, family: ParametricFamily) -> dict[str, float]:
        return {"lambda_": float(np.float64(1.0) / np.float64(0.0))}


class ScriptedComponentEstimationError(ComponentEstimationError):
    """A subclass of the component estimation error for the inheritance test."""


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


def standard_sample() -> tuple[UnivariateFloatArray, UnivariateFloatArray]:
    """The standard sample of this file: cluster ``[4, 5, 6]`` and an outside point."""

    return univariate([4.0, 5.0, 6.0, 100.0]), univariate([1.0, 1.0, 1.0, 0.0])


def assert_estimated(result: PairEvaluation) -> None:
    """Assert the state of an estimated pair."""

    assert result.component is not None
    assert np.isfinite(result.loglikelihood)
    assert result.reason is None


def assert_not_estimated(result: PairEvaluation, reason: str) -> None:
    """Assert the state and the whole reason of a pair that was not estimated."""

    assert result.component is None
    assert result.loglikelihood == -np.inf
    assert result.reason == reason


class TestEstimatorOutput:
    """Handling of the answer, errors and value types of the component estimator."""

    def test_calls_estimator_once_with_the_same_objects_and_does_not_change_arrays(
        self, normal: ParametricFamily
    ) -> None:
        X, h = standard_sample()
        X.flags.writeable = False
        h.flags.writeable = False
        X_before = X.copy()
        h_before = h.copy()
        estimator = ScriptedEstimator({"mu": 5.0, "sigma": 2.0})

        evaluate_pair(X, h, normal, estimator, 1.0)

        assert len(estimator.calls) == 1
        called_X, called_h, called_family = estimator.calls[0]
        assert called_X is X
        assert called_h is h
        assert called_family is normal
        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(h, h_before)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(ComponentEstimationError("scripted reason"), id="base-class"),
            pytest.param(ScriptedComponentEstimationError("scripted reason"), id="subclass"),
        ],
    )
    def test_component_estimation_error_makes_pair_not_estimated(
        self, error: ComponentEstimationError, normal: ParametricFamily
    ) -> None:
        X, h = standard_sample()

        result = evaluate_pair(X, h, normal, ScriptedEstimator(error=error), 1.0)

        assert_not_estimated(result, "scripted reason")

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(NotImplementedError("scripted failure"), id="not-implemented"),
            pytest.param(RuntimeError("scripted failure"), id="runtime"),
            pytest.param(InitializationError("scripted failure"), id="initialization"),
        ],
    )
    def test_other_estimator_errors_pass_through(self, error: BaseException, normal: ParametricFamily) -> None:
        X, h = standard_sample()

        with pytest.raises(type(error), match="^scripted failure$") as excinfo:
            evaluate_pair(X, h, normal, ScriptedEstimator(error=error), 1.0)

        assert excinfo.value is error

    @pytest.mark.parametrize(
        ("make_family", "answer", "message"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL).view(mu=0),
                {"mu": 0, "sigma": 1.0},
                "The component estimator returned parameters ['mu', 'sigma'], "
                "but the free parameters of the family are ['sigma'].",
                id="fixed-parameter-returned",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"mu": 5.0},
                "The component estimator returned parameters ['mu'], "
                "but the free parameters of the family are ['mu', 'sigma'].",
                id="missing",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"mu": 5.0, "scale": 2.0},
                "The component estimator returned parameters ['mu', 'scale'], "
                "but the free parameters of the family are ['mu', 'sigma'].",
                id="other-name",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {},
                "The component estimator returned parameters [], "
                "but the free parameters of the family are ['mu', 'sigma'].",
                id="empty",
            ),
        ],
    )
    def test_rejects_answer_keys_other_than_free_parameters(
        self,
        make_family: Callable[[], ParametricFamily],
        answer: dict[str, object],
        message: str,
    ) -> None:
        X, h = standard_sample()

        with pytest.raises(TypeError, match=rf"^{re.escape(message)}$"):
            evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), 1.0)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("a", id="string"),
            pytest.param(np.array([1.0, 2.0]), id="two-element-array"),
            pytest.param(np.array([1.0]), id="one-element-array"),
            pytest.param(np.array(1.0), id="zero-dimensional-array"),
            pytest.param(None, id="none"),
            pytest.param(1 + 2j, id="complex"),
            pytest.param([1.0], id="list"),
            pytest.param(True, id="bool"),
            pytest.param(np.True_, id="numpy-bool"),
        ],
    )
    def test_rejects_value_that_is_not_a_real_scalar(self, value: object, normal: ParametricFamily) -> None:
        X, h = standard_sample()
        message = (
            f"The component estimator returned a value of type {type(value).__name__} "
            "for parameter 'mu'; a real scalar is expected."
        )

        with pytest.raises(TypeError, match=rf"^{re.escape(message)}$"):
            evaluate_pair(X, h, normal, ScriptedEstimator({"mu": value, "sigma": 2.0}), 1.0)

    @pytest.mark.parametrize(
        "value",
        [pytest.param(np.float32(1.5), id="float32"), pytest.param(np.int64(2), id="int64"), pytest.param(3, id="int")],
    )
    def test_converts_numpy_numbers_and_integers_to_float(
        self, value: int | np.integer | np.floating, normal: ParametricFamily
    ) -> None:
        X = univariate([1.0, 2.0, 3.0])
        h = univariate([1.0, 1.0, 1.0])

        result = evaluate_pair(X, h, normal, ScriptedEstimator({"mu": value, "sigma": 2.0}), 1.0)

        assert_estimated(result)
        assert result.component is not None
        mu = _components.free_parameters(result.component)["mu"]
        assert type(mu) is float
        assert mu == float(value)


class TestNotEstimated:
    """Checks that make a pair impossible before the log-likelihood is computed."""

    def test_rejects_fewer_distinct_values_than_free_parameters(self, normal: ParametricFamily) -> None:
        X = univariate([0.1] * 7 + [5.0, 6.0])
        h = univariate([1.0] * 7 + [0.0, 0.0])

        result = evaluate_pair(X, h, normal, ScriptedEstimator({"mu": 0.1, "sigma": 1.0}), 1.0)

        assert_not_estimated(
            result,
            "the cluster has 1 distinct values, fewer than the 2 free parameters of the family.",
        )

    @pytest.mark.parametrize(
        ("make_family", "X_values", "h_values", "answer"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                [4.0, 4.0, 6.0, 100.0],
                [1.0, 1.0, 1.0, 0.0],
                {"mu": 5.0, "sigma": 2.0},
                id="normal-two-distinct",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                [5.0, 5.0, 5.0, 100.0],
                [1.0, 1.0, 1.0, 0.0],
                {"lambda_": 0.2},
                id="exponential-one-distinct",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL).view(mu=0),
                [0.1] * 7 + [5.0, 6.0],
                [1.0] * 7 + [0.0, 0.0],
                {"sigma": 0.1},
                id="normal-fixed-mu-one-distinct",
            ),
        ],
    )
    def test_accepts_exactly_as_many_distinct_values_as_free_parameters(
        self,
        make_family: Callable[[], ParametricFamily],
        X_values: list[float],
        h_values: list[float],
        answer: dict[str, float],
    ) -> None:
        X = univariate(X_values)
        h = univariate(h_values)

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), 1.0)

        assert_estimated(result)

    @pytest.mark.parametrize(
        ("make_family", "answer", "expected"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"sigma": 1.0, "mu": np.nan},
                {"mu": np.nan, "sigma": 1.0},
                id="mu-nan",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"sigma": 1.0, "mu": np.inf},
                {"mu": np.inf, "sigma": 1.0},
                id="mu-inf",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"sigma": 1.0, "mu": -np.inf},
                {"mu": -np.inf, "sigma": 1.0},
                id="mu-minus-inf",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"mu": 5.0, "sigma": np.nan},
                {"mu": 5.0, "sigma": np.nan},
                id="sigma-nan",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"mu": 5.0, "sigma": np.inf},
                {"mu": 5.0, "sigma": np.inf},
                id="sigma-inf",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                {"mu": 5.0, "sigma": -np.inf},
                {"mu": 5.0, "sigma": -np.inf},
                id="sigma-minus-inf",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                {"lambda_": np.nan},
                {"lambda_": np.nan},
                id="lambda-nan",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                {"lambda_": np.inf},
                {"lambda_": np.inf},
                id="lambda-inf",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                {"lambda_": -np.inf},
                {"lambda_": -np.inf},
                id="lambda-minus-inf",
            ),
        ],
    )
    def test_rejects_non_finite_values_with_parameters_in_family_order(
        self,
        make_family: Callable[[], ParametricFamily],
        answer: dict[str, float],
        expected: dict[str, float],
    ) -> None:
        X, h = standard_sample()

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), 1.0)

        assert_not_estimated(result, f"the estimated parameters are not finite: {expected}.")

    @pytest.mark.parametrize(
        ("make_family", "answer"),
        [
            pytest.param(lambda: family(FamilyName.EXPONENTIAL), {"lambda_": -0.5}, id="negative-rate"),
            pytest.param(lambda: family(FamilyName.NORMAL), {"mu": 5.0, "sigma": 0.0}, id="zero-sigma"),
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM),
                {"lower_bound": 4.0, "upper_bound": 4.0},
                id="equal-bounds",
            ),
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM),
                {"lower_bound": 5.0, "upper_bound": 4.0},
                id="reversed-bounds",
            ),
        ],
    )
    def test_rejects_violated_parametrization_constraints(
        self, make_family: Callable[[], ParametricFamily], answer: dict[str, float]
    ) -> None:
        X, h = standard_sample()

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), 1.0)

        assert result.component is None
        assert result.loglikelihood == -np.inf
        pattern = (
            re.escape(f"the estimated parameters {answer} violate the constraints of the parametrization: ") + ".+"
        )
        assert re.fullmatch(pattern, result.reason)

    def test_rejects_component_too_narrow_before_the_m_step_threshold(self, normal: ParametricFamily) -> None:
        X = univariate([0.1, 0.2, 0.3])
        h = univariate([1.0, 1.0, 1.0])

        result = evaluate_pair(X, h, normal, ScriptedEstimator({"mu": 0.1, "sigma": 1e-17}), 1.0)

        assert_not_estimated(
            result,
            "the interquartile width of the component 0.0 does not exceed 1e-06 of the sample standard deviation 1.0.",
        )

    def test_rejects_width_equal_to_the_relative_threshold(self, uniform: ParametricFamily) -> None:
        X = univariate([1.0, 2.0, 3.0, 100.0])
        h = univariate([1.0, 1.0, 1.0, 0.0])

        result = evaluate_pair(X, h, uniform, ScriptedEstimator({"lower_bound": 0.0, "upper_bound": 4.0}), 2e6)

        assert_not_estimated(
            result,
            "the interquartile width of the component 2.0 does not exceed 1e-06 "
            "of the sample standard deviation 2000000.0.",
        )

    def test_accepts_width_one_float_step_above_the_relative_threshold(self, uniform: ParametricFamily) -> None:
        X = univariate([1.0, 2.0, 3.0, 100.0])
        h = univariate([1.0, 1.0, 1.0, 0.0])
        scale = float(np.nextafter(2e6, 0))

        result = evaluate_pair(X, h, uniform, ScriptedEstimator({"lower_bound": 0.0, "upper_bound": 4.0}), scale)

        assert_estimated(result)

    # Temporary until the M-step tolerance becomes relative (issue #51).
    @pytest.mark.parametrize(
        ("make_family", "X_values", "answer", "scale", "quantity_name", "quantity"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                [1e-5, 2e-5, 3e-5],
                {"mu": 2e-5, "sigma": 1e-5},
                1e-4,
                "variance",
                1e-5**2,
                id="normal-variance",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                [1e-9, 2e-9, 3e-9],
                {"lambda_": 2e9},
                1e-5,
                "mean",
                1 / 2e9,
                id="exponential-mean",
            ),
        ],
    )
    def test_rejects_parameter_at_or_below_the_m_step_tolerance(
        self,
        make_family: Callable[[], ParametricFamily],
        X_values: list[float],
        answer: dict[str, float],
        scale: float,
        quantity_name: str,
        quantity: float,
    ) -> None:
        X = univariate(X_values)
        h = univariate([1.0, 1.0, 1.0])

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), scale)

        assert_not_estimated(
            result,
            f"the {quantity_name} of the component {quantity} does not exceed "
            f"the absolute M-step tolerance {NUMERICAL_TOLERANCE}; rescale the data.",
        )

    # Temporary until the M-step tolerance becomes relative (issue #51).
    @pytest.mark.parametrize(
        ("make_family", "X_values", "answer", "quantity_name", "quantity"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                [4.0, 5.0, 6.0, 100.0],
                {"mu": 5.0, "sigma": 0.5},
                "variance",
                0.25,
                id="normal-variance",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                [1.0, 2.0, 3.0, 100.0],
                {"lambda_": 4.0},
                "mean",
                0.25,
                id="exponential-mean",
            ),
        ],
    )
    def test_rejects_parameter_equal_to_the_module_m_step_tolerance(
        self,
        make_family: Callable[[], ParametricFamily],
        X_values: list[float],
        answer: dict[str, float],
        quantity_name: str,
        quantity: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_validation, "NUMERICAL_TOLERANCE", 0.25)
        X = univariate(X_values)
        h = univariate([1.0, 1.0, 1.0, 0.0])

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), 1.0)

        assert_not_estimated(
            result,
            f"the {quantity_name} of the component {quantity} does not exceed "
            f"the absolute M-step tolerance {_validation.NUMERICAL_TOLERANCE}; rescale the data.",
        )

    # Temporary until the M-step tolerance becomes relative (issue #51).
    @pytest.mark.parametrize(
        ("make_family", "X_values", "answer"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL),
                [4.0, 5.0, 6.0, 100.0],
                {"mu": 5.0, "sigma": float(np.nextafter(0.5, 1))},
                id="normal-variance",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL),
                [1.0, 2.0, 3.0, 100.0],
                {"lambda_": float(np.nextafter(4.0, 0))},
                id="exponential-mean",
            ),
        ],
    )
    def test_accepts_parameter_one_float_step_above_the_module_m_step_tolerance(
        self,
        make_family: Callable[[], ParametricFamily],
        X_values: list[float],
        answer: dict[str, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_validation, "NUMERICAL_TOLERANCE", 0.25)
        X = univariate(X_values)
        h = univariate([1.0, 1.0, 1.0, 0.0])

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), 1.0)

        assert_estimated(result)

    # Temporary until the M-step tolerance becomes relative (issue #51).
    @pytest.mark.parametrize(
        ("make_family", "X_values", "answer", "scale"),
        [
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM),
                [0.5e-10, 1e-10, 1.5e-10],
                {"lower_bound": 0.0, "upper_bound": 2e-10},
                1e-5,
                id="uniform",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL).view(sigma=1e-5),
                [0.0, 1e-5, 2e-5],
                {"mu": 1e-5},
                1e-4,
                id="normal-fixed-sigma",
            ),
            pytest.param(
                lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar"),
                [0.0, 1e-5, 2e-5],
                {"mu": 1e-5, "var": 1e-10},
                1e-4,
                id="normal-meanVar",
            ),
        ],
    )
    def test_m_step_threshold_does_not_apply_to_other_cases(
        self,
        make_family: Callable[[], ParametricFamily],
        X_values: list[float],
        answer: dict[str, float],
        scale: float,
    ) -> None:
        X = univariate(X_values)
        h = univariate([1.0, 1.0, 1.0])

        result = evaluate_pair(X, h, make_family(), ScriptedEstimator(answer), scale)

        assert_estimated(result)


class TestLoglikelihood:
    """Component and cluster log-likelihood of an evaluated pair."""

    def test_estimated_pair_contains_component_parameters_and_cluster_loglikelihood(
        self, normal: ParametricFamily
    ) -> None:
        X, h = standard_sample()

        result = evaluate_pair(X, h, normal, ScriptedEstimator({"mu": 5.0, "sigma": 2.0}), 1.0)

        assert isinstance(result, PairEvaluation)
        assert result.component is not None
        assert _components.component_family_key(result.component) == _components.family_key(normal)
        assert _components.free_parameters(result.component) == {"mu": 5.0, "sigma": 2.0}
        assert result.loglikelihood == pytest.approx(-5.086257141293855, rel=1e-12)
        assert type(result.loglikelihood) is float
        assert result.reason is None

    def test_only_cluster_points_enter_the_loglikelihood(self, expon: ParametricFamily) -> None:
        X = univariate([-7.0, 1.0, 2.0])
        h = univariate([0.0, 1.0, 1.0])

        result = evaluate_pair(X, h, expon, ScriptedEstimator({"lambda_": 0.5}), 1.0)

        assert result.loglikelihood == pytest.approx(-2.886294361119891, rel=1e-12)
        assert result.reason is None

    def test_cluster_point_outside_support_keeps_component_and_marks_pair_as_not_covering(
        self, expon: ParametricFamily
    ) -> None:
        X = univariate([-1.0, 5.0, 100.0])
        h = univariate([1.0, 1.0, 0.0])

        result = evaluate_pair(X, h, expon, ScriptedEstimator({"lambda_": 0.5}), 1.0)

        assert result.component is not None
        assert _components.free_parameters(result.component) == {"lambda_": 0.5}
        assert result.loglikelihood == -np.inf
        assert (
            result.reason == "1 of 2 points of the cluster are outside the support of the component, for example -1.0."
        )

    def test_reason_has_at_most_three_cluster_examples_in_sample_order(self, expon: ParametricFamily) -> None:
        X = univariate([-9.0, -3.0, 2.0, -1.0, -5.0, -4.0, -2.0])
        h = univariate([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        result = evaluate_pair(X, h, expon, ScriptedEstimator({"lambda_": 0.5}), 1.0)

        assert result.reason == (
            "5 of 6 points of the cluster are outside the support of the component, for example -3.0, -1.0, -5.0."
        )

    def test_points_on_both_sides_of_a_finite_support_are_reported(self, uniform: ParametricFamily) -> None:
        X = univariate([-1.0, 2.0, 5.0, 100.0])
        h = univariate([1.0, 1.0, 1.0, 0.0])

        result = evaluate_pair(X, h, uniform, ScriptedEstimator({"lower_bound": 0.0, "upper_bound": 4.0}), 1.0)

        assert result.component is not None
        assert result.loglikelihood == -np.inf
        assert result.reason == (
            "2 of 3 points of the cluster are outside the support of the component, for example -1.0, 5.0."
        )

    def test_reports_non_finite_loglikelihood_without_points_outside_support(self, normal: ParametricFamily) -> None:
        X = univariate([1e154, -1e154, 1.1e154, -1.1e154])
        h = univariate([1.0, 1.0, 1.0, 1.0])

        result = evaluate_pair(X, h, normal, ScriptedEstimator({"mu": 0.0, "sigma": 1.0}), 1.0)

        assert result.component is not None
        assert result.loglikelihood == -np.inf
        assert result.reason == "the log-likelihood of the cluster is -inf."

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("overflow-of-the-sum", id="overflow-of-the-sum"),
            pytest.param("division-inside-the-estimator", id="division-inside-the-estimator"),
        ],
    )
    def test_numpy_warnings_do_not_escape(self, case: str, normal: ParametricFamily, expon: ParametricFamily) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            if case == "overflow-of-the-sum":
                X = univariate([1e154, -1e154, 1.1e154, -1.1e154])
                h = univariate([1.0, 1.0, 1.0, 1.0])

                result = evaluate_pair(X, h, normal, ScriptedEstimator({"mu": 0.0, "sigma": 1.0}), 1.0)

                assert result.component is not None
            else:
                X, h = standard_sample()

                result = evaluate_pair(X, h, expon, NumpyDivisionEstimator(), 1.0)

                assert_not_estimated(result, "the estimated parameters are not finite: {'lambda_': inf}.")
