"""Integration tests for the registered Method of Moments strategies."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import moments as moments_module
from pysatl_mpest.estimators.iterative._strategies import moments_strategy, q_function_strategy
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.families import family
from tests.helpers.moments import weighted_moments
from tests.helpers.pipeline import make_state, q_block


def moments_block(component_id: int = 0, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.MOMENTS)


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def exponential(lambda_: float) -> ParametricFamilyDistribution:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


def two_component_state(component: ParametricFamilyDistribution | None = None) -> PipelineState:
    components = [component, component] if component is not None else [normal(0.0, 1.0), normal(0.0, 1.0)]
    mixture = MixtureModel(components)
    return make_state(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]),
        mixture=mixture,
    )


def one_component_state(
    X: np.ndarray, H: np.ndarray | None, component: ParametricFamilyDistribution | None = None
) -> PipelineState:
    mixture = MixtureModel([component if component is not None else normal(0.0, 1.0)])
    return make_state(X=X, H=H, mixture=mixture)


def run_moments(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock
) -> tuple[int, dict[str, float]]:
    return moments_strategy(component, state, block, optimizer=ScipyNelderMead())


@st.composite
def normal_sample(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    true_mu = draw(st.floats(min_value=-50, max_value=50))
    true_sigma = draw(st.floats(min_value=0.5, max_value=20))
    rng = np.random.default_rng(42)
    return rng.normal(loc=true_mu, scale=true_sigma, size=10000), true_mu, true_sigma


@st.composite
def normal_sample_with_random_weights(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, float, float]:
    true_mu = draw(st.floats(min_value=-10.0, max_value=10.0))
    true_sigma = draw(st.floats(min_value=0.5, max_value=5.0))
    rng = np.random.default_rng(42)
    X = rng.normal(loc=true_mu, scale=true_sigma, size=15000)
    return X, rng.uniform(low=0.1, high=1.0, size=15000), true_mu, true_sigma


@st.composite
def exponential_sample(draw: st.DrawFn) -> tuple[np.ndarray, float]:
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    rng = np.random.default_rng(42)
    return rng.exponential(scale=1.0 / true_rate, size=10000), true_rate


class TestSharedContracts:
    @pytest.mark.parametrize(
        ("component", "params", "expected"),
        [
            pytest.param(normal(0.0, 1.0), None, {"mu", "sigma"}, id="normal-meanStd-all"),
            pytest.param(normal(0.0, 1.0), {"mu", "sigma"}, {"mu", "sigma"}, id="normal-meanStd-explicit-all"),
            pytest.param(normal(0.0, 1.0), {"mu"}, {"mu"}, id="normal-meanStd-mean"),
            pytest.param(normal(0.0, 1.0), {"sigma"}, {"sigma"}, id="normal-meanStd-standard-deviation"),
            pytest.param(normal(0.0, 1.0), set(), set(), id="normal-meanStd-empty"),
            pytest.param(exponential(1.0), None, {"lambda_"}, id="exponential-rate-all"),
            pytest.param(exponential(1.0), {"lambda_"}, {"lambda_"}, id="exponential-rate-explicit"),
            pytest.param(exponential(1.0), set(), set(), id="exponential-rate-empty"),
        ],
    )
    def test_updates_only_the_requested_parameters(
        self, component: ParametricFamilyDistribution, params: set[str] | None, expected: set[str]
    ) -> None:
        _, new_params = run_moments(component, two_component_state(component), moments_block(params=params))

        assert set(new_params) == expected

    @pytest.mark.parametrize(
        ("component", "unknown_param", "message"),
        [
            pytest.param(
                normal(0.0, 1.0),
                "var",
                r"\['var'\] are not parameters of the 'meanStd' parametrization",
                id="normal-meanStd",
            ),
            pytest.param(
                exponential(1.0),
                "beta",
                r"\['beta'\] are not parameters of the 'rate' parametrization",
                id="exponential-rate",
            ),
        ],
    )
    def test_rejects_unknown_parameter_names(
        self, component: ParametricFamilyDistribution, unknown_param: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            run_moments(component, two_component_state(component), moments_block(params={unknown_param}))

    @pytest.mark.parametrize(
        "component",
        [pytest.param(normal(0.0, 1.0), id="normal-meanStd"), pytest.param(exponential(1.0), id="exponential-rate")],
    )
    def test_requires_computed_responsibilities(self, component: ParametricFamilyDistribution) -> None:
        state = one_component_state(np.array([1.0]), H=None, component=component)

        with pytest.raises(ValueError, match=r"^Responsibility matrix H is not computed\.$"):
            run_moments(component, state, moments_block())

    def test_rejects_requested_fixed_parameters(self) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

        with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
            run_moments(fixed_mu, two_component_state(fixed_mu), moments_block(params={"mu", "sigma"}))

    @pytest.mark.parametrize(
        "component",
        [pytest.param(normal(0.0, 1.0), id="normal-meanStd"), pytest.param(exponential(1.0), id="exponential-rate")],
    )
    def test_skips_updates_for_negligible_responsibility(self, component: ParametricFamilyDistribution) -> None:
        state = two_component_state(component)
        assert state.H is not None
        state.H.fill(1e-10)

        assert run_moments(component, state, moments_block()) == (0, {})


class TestNormalFormula:
    """Check weighted-moment updates for normal/meanStd."""

    def test_returns_the_weighted_mean_and_standard_deviation(self) -> None:
        state = one_component_state(np.array([10.0, 20.0]), np.array([[0.2], [0.8]]))

        component_id, new_params = run_moments(normal(0.0, 1.0), state, moments_block())

        assert component_id == 0
        assert new_params == pytest.approx({"mu": 18.0, "sigma": 4.0})
        assert all(type(value) is float for value in new_params.values())

    def test_updates_only_the_standard_deviation_for_a_fixed_mean(self) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

        _, new_params = run_moments(fixed_mu, two_component_state(fixed_mu), moments_block())

        assert set(new_params) == {"sigma"}

    def test_updates_only_the_mean_for_a_fixed_standard_deviation(self) -> None:
        fixed_sigma = family(FamilyName.NORMAL).view(sigma=2.0)(mu=0.0)

        _, new_params = run_moments(fixed_sigma, two_component_state(fixed_sigma), moments_block())

        assert set(new_params) == {"mu"}

    def test_measures_deviation_around_the_fixed_mean(self) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)
        state = two_component_state(fixed_mu)
        _, std_around_fixed = weighted_moments(state, mu=0.0)

        _, new_params = run_moments(fixed_mu, state, moments_block())

        assert new_params == pytest.approx({"sigma": std_around_fixed})

    def test_keeps_the_standard_deviation_when_all_observations_coincide(self) -> None:
        component = normal(0.0, 2.5)
        state = one_component_state(np.array([10.0, 10.0, 10.0]), np.ones((3, 1)), component)

        _, new_params = run_moments(component, state, moments_block())

        assert new_params == {"mu": 10.0, "sigma": 2.5}

    def test_keeps_the_standard_deviation_when_observations_equal_the_fixed_mean(self) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=10.0)(sigma=2.5)
        state = one_component_state(np.array([10.0, 10.0, 10.0]), np.ones((3, 1)), fixed_mu)

        _, new_params = run_moments(fixed_mu, state, moments_block())

        assert new_params == {"sigma": 2.5}


@settings(max_examples=50)
@given(data=normal_sample())
def test_recovers_parameters_from_ideal_data(data: tuple[np.ndarray, float, float]) -> None:
    X, true_mu, true_sigma = data
    H_j = np.ones_like(X)
    state = make_state(
        X=X,
        H=np.vstack([H_j, np.zeros_like(H_j)]).T,
        mixture=MixtureModel([normal(-999.0, 0.001), normal(0.0, 1.0)]),
    )

    _, new_params = run_moments(normal(-999.0, 0.001), state, moments_block())

    assert new_params["mu"] == pytest.approx(true_mu, abs=1.0)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.1)


@settings(max_examples=30, deadline=None)
@given(data=normal_sample_with_random_weights())
def test_converges_to_parameters_with_random_weights(data: tuple[np.ndarray, np.ndarray, float, float]) -> None:
    X, weights, true_mu, true_sigma = data
    state = make_state(
        X=X,
        H=np.vstack([weights, np.zeros_like(weights)]).T,
        mixture=MixtureModel([normal(true_mu + 10, true_sigma + 5), normal(0.0, 1.0)]),
    )

    _, new_params = run_moments(normal(true_mu + 10, true_sigma + 5), state, moments_block())

    assert new_params["mu"] == pytest.approx(true_mu, abs=0.2, rel=0.1)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.15)


class TestExponentialFormula:
    """Check weighted-moment updates for exponential/rate."""

    def test_returns_the_inverse_of_the_weighted_mean(self) -> None:
        state = two_component_state(exponential(1.0))
        assert state.H is not None
        H_j = state.H[:, 0]
        expected = H_j.sum() / np.dot(H_j, state.X)

        component_id, new_params = run_moments(exponential(1.0), state, moments_block())

        assert component_id == 0
        assert new_params == pytest.approx({"lambda_": expected})
        assert type(new_params["lambda_"]) is float

    @pytest.mark.parametrize("values", [[0.0, 0.0], [-1.0, -2.0]], ids=["zero mean", "negative mean"])
    def test_keeps_the_rate_when_the_weighted_mean_is_not_positive(self, values: list[float]) -> None:
        component = exponential(2.5)
        state = one_component_state(np.array(values), np.ones((2, 1)), component)

        _, new_params = run_moments(component, state, moments_block())

        assert new_params == {"lambda_": 2.5}


@settings(max_examples=50)
@given(data=exponential_sample())
def test_recovers_the_rate_from_ideal_data(data: tuple[np.ndarray, float]) -> None:
    X, true_rate = data
    H_j = np.ones_like(X)
    state = make_state(
        X=X,
        H=np.vstack([H_j, np.zeros_like(H_j)]).T,
        mixture=MixtureModel([exponential(0.001), exponential(1.0)]),
    )

    _, new_params = run_moments(exponential(0.001), state, moments_block())

    assert new_params["lambda_"] == pytest.approx(true_rate, rel=0.1)


class TestUnregisteredPairs:
    @pytest.mark.parametrize(
        ("make_component", "message"),
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0),
                "'meanVar' parametrization of the 'Normal' family",
                id="normal-meanVar",
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL)(parametrization_name="scale", beta=1.0),
                "'scale' parametrization of the 'Exponential' family",
                id="exponential-scale",
            ),
            pytest.param(
                lambda: family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0),
                "'standard' parametrization of the 'ContinuousUniform' family",
                id="uniform",
            ),
        ],
    )
    def test_raises_not_implemented_error(
        self, make_component: Callable[[], ParametricFamilyDistribution], message: str
    ) -> None:
        component = make_component()
        state = one_component_state(np.array([0.5]), np.ones((1, 1)), component)
        block = moments_block()

        with pytest.raises(NotImplementedError, match=f"Moments strategy is not implemented for the {message}"):
            run_moments(component, state, block)


class TestEstimateComparison:
    @pytest.mark.parametrize(
        ("component", "component_id"),
        [
            pytest.param(normal(0.0, 1.0), 0, id="normal-component-0"),
            pytest.param(normal(0.0, 1.0), 1, id="normal-component-1"),
            pytest.param(exponential(1.0), 1, id="exponential-component-1"),
        ],
    )
    def test_estimates_match_for_registered_components(
        self, component: ParametricFamilyDistribution, component_id: int
    ) -> None:
        state = two_component_state(component)
        _, by_moments = run_moments(component, state, moments_block(component_id=component_id))
        _, by_q_function = q_function_strategy(
            component, state, q_block(component_id=component_id), optimizer=ScipyNelderMead()
        )

        assert by_moments == pytest.approx(by_q_function)


class TestDispatch:
    def test_registered_keys_include_both_analytical_pairs(self) -> None:
        assert sorted(moments_strategy.registry) == [("Exponential", "rate"), ("Normal", "meanStd")]

    @pytest.mark.parametrize(
        ("make_component", "key"),
        [
            pytest.param(lambda: normal(0.0, 1.0), ("Normal", "meanStd"), id="normal-meanStd"),
            pytest.param(
                lambda: family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0),
                ("Normal", "meanStd"),
                id="normal-meanStd-view",
            ),
            pytest.param(lambda: exponential(1.0), ("Exponential", "rate"), id="exponential-rate"),
        ],
    )
    def test_selects_the_registered_implementation(
        self, make_component: Callable[[], ParametricFamilyDistribution], key: tuple[str, str]
    ) -> None:
        selected = moments_strategy.dispatch(make_component())

        assert selected is moments_strategy.registry[key]
        assert selected is not moments_module._not_implemented

    @pytest.mark.parametrize(
        "make_component",
        [
            pytest.param(
                lambda: family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0), id="normal-meanVar"
            ),
            pytest.param(
                lambda: family(FamilyName.EXPONENTIAL)(parametrization_name="scale", beta=1.0), id="exponential-scale"
            ),
            pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0), id="uniform"),
        ],
    )
    def test_selects_the_default_implementation_for_unregistered_pairs(
        self, make_component: Callable[[], ParametricFamilyDistribution]
    ) -> None:
        assert moments_strategy.dispatch(make_component()) is moments_module._not_implemented

    def test_strategy_has_its_own_name_and_docstring(self) -> None:
        """Strategy has its own name and docstring."""
        doc = moments_strategy.__doc__ or ""

        assert moments_strategy.__name__ == "moments_strategy"
        for token in ("Normal", "meanStd", "Exponential", "rate", "NotImplementedError"):
            assert token in doc


@pytest.fixture
def coarse_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moments_module, "NUMERICAL_TOLERANCE", 0.25)


def run_at_threshold(
    component: ParametricFamilyDistribution, X: list[float], H: list[list[float]], params: set[str] | None = None
) -> dict[str, float]:
    state = one_component_state(np.array(X), np.array(H), component)
    _, new_params = run_moments(component, state, moments_block(params=params))
    return new_params


class TestResponsibilityThreshold:
    """Check the moments strategy's numerical responsibility threshold."""

    @pytest.mark.parametrize(
        "component",
        [pytest.param(exponential(2.5), id="exponential-rate"), pytest.param(normal(0.0, 2.0), id="normal-meanStd")],
    )
    def test_skips_updates_at_the_tolerance(
        self, coarse_tolerance: None, component: ParametricFamilyDistribution
    ) -> None:
        assert run_at_threshold(component, [1.0], [[0.25]]) == {}

    @pytest.mark.parametrize(
        "component",
        [pytest.param(exponential(2.5), id="exponential-rate"), pytest.param(normal(0.0, 2.0), id="normal-meanStd")],
    )
    def test_updates_parameters_above_the_tolerance(
        self, coarse_tolerance: None, component: ParametricFamilyDistribution
    ) -> None:
        assert run_at_threshold(component, [1.0], [[0.5]]) != {}


class TestExponentialThreshold:
    """Check the moments strategy's exponential mean threshold."""

    def test_keeps_the_rate_at_the_tolerance(self, coarse_tolerance: None) -> None:
        assert run_at_threshold(exponential(2.5), [0.25], [[1.0]]) == {"lambda_": 2.5}

    def test_updates_the_rate_above_the_tolerance(self, coarse_tolerance: None) -> None:
        assert run_at_threshold(exponential(2.5), [0.5], [[1.0]]) == {"lambda_": 2.0}


class TestNormalVarianceThreshold:
    """Check the moments strategy's normal variance threshold."""

    def test_keeps_the_standard_deviation_at_the_tolerance(self, coarse_tolerance: None) -> None:
        # The mean is 0.5 and the variance is exactly 0.25.
        result = run_at_threshold(normal(0.0, 2.0), [0.0, 1.0], [[0.5], [0.5]])

        assert result == {"mu": 0.5, "sigma": 2.0}

    def test_updates_the_standard_deviation_above_the_tolerance(self, coarse_tolerance: None) -> None:
        # The mean is 1 and the variance is 1.
        result = run_at_threshold(normal(0.0, 2.0), [0.0, 2.0], [[0.5], [0.5]])

        assert result == {"mu": 1.0, "sigma": 1.0}
