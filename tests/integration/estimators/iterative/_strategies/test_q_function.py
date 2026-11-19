"""
Integration tests for analytical and numerical Q-function strategies.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function as q_function_module
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead, ScipyPowell

from tests.helpers.families import family
from tests.helpers.moments import weighted_moments
from tests.helpers.pipeline import make_state, q_block

TOLERANCE = 0.25


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    """Create a normal component in the meanStd parametrization.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.

    Returns
    -------
    ParametricFamilyDistribution
        The component.
    """
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def exponential(lambda_: float) -> ParametricFamilyDistribution:
    """Create an exponential component in the rate parametrization.

    Parameters
    ----------
    lambda_ : float
        Rate.

    Returns
    -------
    ParametricFamilyDistribution
        The component.
    """
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


@pytest.fixture(params=["normal-meanStd", "exponential-rate"])
def analytical_component(request: pytest.FixtureRequest) -> ParametricFamilyDistribution:
    """Provide each registered analytical pair with representative parameters."""
    return normal(0.0, 1.0) if request.param == "normal-meanStd" else exponential(1.0)


def _analytical_state(component: ParametricFamilyDistribution) -> PipelineState:
    return make_state(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]),
        mixture=MixtureModel([component] * 2),
    )


@pytest.fixture
def normal_state() -> PipelineState:
    """Provide the normal sample and two matching components."""
    return _analytical_state(normal(0.0, 1.0))


@pytest.fixture
def exponential_state() -> PipelineState:
    """Provide the exponential sample and two matching components."""
    return _analytical_state(exponential(1.0))


@pytest.fixture
def coarse_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set only the Q-function threshold to the binary-exact value 0.25."""
    monkeypatch.setattr(q_function_module, "NUMERICAL_TOLERANCE", TOLERANCE)


def _run_threshold(
    component: ParametricFamilyDistribution,
    X: Sequence[float],
    H: Sequence[Sequence[float]],
) -> dict[str, float]:
    state = make_state(np.array(X), np.array(H), mixture=MixtureModel([component]))
    return q_function_strategy(component, state, q_block(), ScipyNelderMead())[1]


def sample_state(true_mu: float, true_var: float, size: int = 2000) -> PipelineState:
    """Build a seeded normal sample with unit responsibility for the first component.

    Parameters
    ----------
    true_mu : float
        Generating mean.
    true_var : float
        Generating variance.
    size : int, default=2000
        Number of observations.

    Returns
    -------
    PipelineState
        The sample, responsibilities and a matching two-component mixture.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(true_mu, np.sqrt(true_var), size)
    return make_state(
        X=X,
        H=np.column_stack([np.ones_like(X), np.zeros_like(X)]),
        mixture=MixtureModel([family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)] * 2),
    )


def weighted_state(size: int = 3000) -> PipelineState:
    """Build seeded positive observations with nonuniform responsibilities.

    Parameters
    ----------
    size : int, default=3000
        Number of observations.

    Returns
    -------
    PipelineState
        The sample, responsibilities and a matching two-component mixture.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 1.5, size)
    H_j = rng.random(size)
    H_j[::7] = 0.0
    return make_state(
        X=np.abs(X) + 0.1,
        H=np.column_stack([H_j, 1.0 - H_j]),
        mixture=MixtureModel([family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)] * 2),
    )


@st.composite
def normal_sample(draw: st.DrawFn) -> tuple[NDArray[np.float64], float, float]:
    """Draw normal parameters and generate their seeded sample.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.

    Returns
    -------
    tuple[NDArray[np.float64], float, float]
        Sample, generating mean and generating standard deviation.
    """
    true_mu = draw(st.floats(min_value=-50, max_value=50))
    true_sigma = draw(st.floats(min_value=0.5, max_value=20))
    rng = np.random.default_rng(42)
    return rng.normal(loc=true_mu, scale=true_sigma, size=10000), true_mu, true_sigma


@st.composite
def exponential_sample(draw: st.DrawFn) -> tuple[NDArray[np.float64], float]:
    """Draw an exponential rate and generate its seeded sample.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.

    Returns
    -------
    tuple[NDArray[np.float64], float]
        Sample and generating rate.
    """
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    rng = np.random.default_rng(42)
    return rng.exponential(scale=1.0 / true_rate, size=10000), true_rate


ANALYTICAL = [
    pytest.param(lambda: family(FamilyName.NORMAL)(mu=0.0, sigma=1.0), ("Normal", "meanStd"), id="normal-meanStd"),
    pytest.param(
        lambda: family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0), ("Normal", "meanStd"), id="normal-meanStd-view"
    ),
    pytest.param(lambda: family(FamilyName.EXPONENTIAL)(lambda_=1.0), ("Exponential", "rate"), id="exponential-rate"),
]


GENERIC = [
    pytest.param(
        lambda: family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0), id="normal-meanVar"
    ),
    pytest.param(
        lambda: family(FamilyName.EXPONENTIAL)(parametrization_name="scale", beta=1.0), id="exponential-scale"
    ),
    pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0), id="uniform"),
]


class TestDispatch:
    """Dispatch for registered and fallback pairs."""

    def test_registers_only_the_supported_analytical_pairs(self) -> None:
        """Registers only the supported analytical pairs."""
        assert sorted(q_function_strategy.registry) == [("Exponential", "rate"), ("Normal", "meanStd")]

    @pytest.mark.parametrize(("make_component", "key"), ANALYTICAL)
    def test_selects_the_registered_implementation_including_views(
        self, make_component: Callable[[], ParametricFamilyDistribution], key: tuple[str, str]
    ) -> None:
        """Selects the registered implementation including views."""
        selected = q_function_strategy.dispatch(make_component())

        assert selected is q_function_strategy.registry[key]
        assert selected is not q_function_module._numerical_q_function

    @pytest.mark.parametrize("make_component", GENERIC)
    def test_selects_the_default_for_unregistered_pairs(
        self, make_component: Callable[[], ParametricFamilyDistribution]
    ) -> None:
        """Selects the default for unregistered pairs."""
        assert q_function_strategy.dispatch(make_component()) is q_function_module._numerical_q_function

    def test_strategy_has_its_own_name_and_docstring(self) -> None:
        """Strategy has its own name and docstring."""
        doc = q_function_strategy.__doc__ or ""

        assert q_function_strategy.__name__ == "q_function_strategy"
        for token in ("Normal", "meanStd", "Exponential", "rate"):
            assert token in doc


class TestNormalFormula:
    """Analytical normal mean and standard deviation."""

    def test_returns_the_weighted_mean_and_standard_deviation(self, normal_state: PipelineState) -> None:
        """Returns the weighted mean and standard deviation."""
        mean, std = weighted_moments(normal_state)

        component_id, new_params = q_function_strategy(
            normal(0.0, 1.0), normal_state, q_block(), optimizer=ScipyNelderMead()
        )

        assert component_id == 0
        assert new_params == pytest.approx({"mu": mean, "sigma": std})
        assert all(type(value) is float for value in new_params.values())

    def test_uses_the_column_of_the_block_component(self, normal_state: PipelineState) -> None:
        """Uses the column of the block component."""
        mean, std = weighted_moments(normal_state, component_id=1)

        _, new_params = q_function_strategy(
            normal(0.0, 1.0), normal_state, q_block(component_id=1), optimizer=ScipyNelderMead()
        )

        assert new_params == pytest.approx({"mu": mean, "sigma": std})

    def test_standard_deviation_is_taken_around_the_fixed_mean(self, normal_state: PipelineState) -> None:
        """Standard deviation is taken around the fixed mean."""
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)
        _, std_around_fixed = weighted_moments(normal_state, mu=0.0)
        _, std_around_mean = weighted_moments(normal_state)

        _, new_params = q_function_strategy(fixed_mu, normal_state, q_block(), optimizer=ScipyNelderMead())

        assert new_params["sigma"] == pytest.approx(std_around_fixed)
        assert new_params["sigma"] != pytest.approx(std_around_mean)

    def test_standard_deviation_is_taken_around_the_new_mean(self, normal_state: PipelineState) -> None:
        """Standard deviation is taken around the new mean."""
        mean, std = weighted_moments(normal_state)

        _, new_params = q_function_strategy(normal(10.0, 1.0), normal_state, q_block(), optimizer=ScipyNelderMead())

        assert new_params == pytest.approx({"mu": mean, "sigma": std})

    def test_standard_deviation_uses_the_current_mean_if_only_sigma_is_requested(
        self, normal_state: PipelineState
    ) -> None:
        """Standard deviation uses the current mean if only sigma is requested."""
        _, std_around_current = weighted_moments(normal_state, mu=10.0)

        _, new_params = q_function_strategy(
            normal(10.0, 1.0), normal_state, q_block(params={"sigma"}), optimizer=ScipyNelderMead()
        )

        assert new_params == pytest.approx({"sigma": std_around_current})

    def test_keeps_the_standard_deviation_if_the_variance_is_zero(self) -> None:
        """Keeps the standard deviation if the variance is zero."""
        normal_state = make_state(
            X=np.array([10.0, 10.0]), H=np.array([[1.0, 0.0], [1.0, 0.0]]), mixture=MixtureModel([normal(0.0, 1.0)] * 2)
        )
        fixed_mu = family(FamilyName.NORMAL).view(mu=10.0)(sigma=2.5)

        _, new_params = q_function_strategy(fixed_mu, normal_state, q_block(), optimizer=ScipyNelderMead())

        assert new_params == {"sigma": 2.5}


@settings(max_examples=50, deadline=None)
@given(data=normal_sample())
def test_recovers_the_true_parameters_on_ideal_data(data: tuple[NDArray[np.float64], float, float]) -> None:
    """Recovers the true parameters on ideal data."""
    X, true_mu, true_sigma = data
    H_j = np.ones_like(X)
    normal_state = make_state(
        X=X, H=np.vstack([H_j, np.zeros_like(H_j)]).T, mixture=MixtureModel([normal(0.0, 1.0)] * 2)
    )

    _, new_params = q_function_strategy(normal(-999.0, 0.001), normal_state, q_block(), optimizer=ScipyNelderMead())

    assert new_params["mu"] == pytest.approx(true_mu, abs=1.0)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.1)


class TestExponentialFormula:
    """Analytical exponential rate."""

    def test_returns_the_inverse_of_the_weighted_mean(self, exponential_state: PipelineState) -> None:
        """Returns the inverse of the weighted mean."""
        assert exponential_state.H is not None
        H_j = exponential_state.H[:, 0]
        expected = H_j.sum() / np.dot(H_j, exponential_state.X)

        component_id, new_params = q_function_strategy(
            exponential(1.0), exponential_state, q_block(), optimizer=ScipyNelderMead()
        )

        assert component_id == 0
        assert new_params == pytest.approx({"lambda_": expected})
        assert type(new_params["lambda_"]) is float

    def test_uses_the_column_of_the_block_component(self, exponential_state: PipelineState) -> None:
        """Uses the column of the block component."""
        assert exponential_state.H is not None
        H_j = exponential_state.H[:, 1]

        _, new_params = q_function_strategy(
            exponential(1.0), exponential_state, q_block(component_id=1), optimizer=ScipyNelderMead()
        )

        assert new_params["lambda_"] == pytest.approx(H_j.sum() / np.dot(H_j, exponential_state.X))

    def test_keeps_the_rate_if_the_weighted_mean_is_zero(self) -> None:
        """Keeps the rate if the weighted mean is zero."""
        exponential_state = make_state(
            X=np.array([0.0]), H=np.array([[1.0, 0.0]]), mixture=MixtureModel([exponential(1.0)] * 2)
        )

        _, new_params = q_function_strategy(exponential(2.5), exponential_state, q_block(), optimizer=ScipyNelderMead())

        assert new_params == {"lambda_": 2.5}


@settings(max_examples=50)
@given(data=exponential_sample())
def test_recovers_the_true_rate_on_ideal_data(data: tuple[NDArray[np.float64], float]) -> None:
    """Recovers the true rate on ideal data."""
    X, true_rate = data
    H_j = np.ones_like(X)
    exponential_state = make_state(
        X=X, H=np.vstack([H_j, np.zeros_like(H_j)]).T, mixture=MixtureModel([exponential(1.0)] * 2)
    )

    _, new_params = q_function_strategy(exponential(0.001), exponential_state, q_block(), optimizer=ScipyNelderMead())

    assert new_params["lambda_"] == pytest.approx(true_rate, rel=0.05)


class TestAnalyticalContracts:
    """Contracts shared by the registered analytical pairs."""

    def test_raises_if_h_is_not_computed(self, analytical_component: ParametricFamilyDistribution) -> None:
        """Reject missing responsibilities for each registered pair."""
        state = make_state(X=np.array([1.0]), H=None, mixture=MixtureModel([analytical_component]))

        with pytest.raises(ValueError, match=r"^Responsibility matrix H is not computed\.$"):
            q_function_strategy(analytical_component, state, q_block(), ScipyNelderMead())

    @pytest.mark.parametrize(
        ("make_component", "params", "expected"),
        [
            pytest.param(lambda: normal(0.0, 1.0), None, {"mu", "sigma"}, id="normal-meanStd-all"),
            pytest.param(lambda: normal(0.0, 1.0), {"mu", "sigma"}, {"mu", "sigma"}, id="normal-meanStd-both"),
            pytest.param(lambda: normal(0.0, 1.0), {"mu"}, {"mu"}, id="normal-meanStd-mu"),
            pytest.param(lambda: normal(0.0, 1.0), {"sigma"}, {"sigma"}, id="normal-meanStd-sigma"),
            pytest.param(lambda: exponential(1.0), None, {"lambda_"}, id="exponential-rate-all"),
            pytest.param(lambda: exponential(1.0), {"lambda_"}, {"lambda_"}, id="exponential-rate-lambda"),
        ],
    )
    def test_updates_the_requested_parameters(
        self,
        make_component: Callable[[], ParametricFamilyDistribution],
        params: set[str] | None,
        expected: set[str],
    ) -> None:
        """Return only the keys requested for each analytical pair."""
        component = make_component()
        state = _analytical_state(component)

        _, new_params = q_function_strategy(component, state, q_block(params=params), ScipyNelderMead())

        assert set(new_params) == expected

    @pytest.mark.parametrize(
        "make_component",
        [
            pytest.param(lambda: normal(0.0, 1.0), id="normal-meanStd-empty"),
            pytest.param(lambda: exponential(1.0), id="exponential-rate-empty"),
        ],
    )
    def test_updates_nothing_when_nothing_is_requested(
        self, make_component: Callable[[], ParametricFamilyDistribution]
    ) -> None:
        """Return the component id and no parameters for an empty request."""
        component = make_component()
        state = _analytical_state(component)

        result = q_function_strategy(component, state, q_block(params=set()), ScipyNelderMead())

        assert result == (0, {})

    @pytest.mark.parametrize(
        ("make_component", "params", "message"),
        [
            pytest.param(
                lambda: normal(0.0, 1.0),
                {"var"},
                r"\['var'\] are not parameters of the 'meanStd' parametrization",
                id="normal-meanStd-var",
            ),
            pytest.param(
                lambda: exponential(1.0),
                {"beta"},
                r"\['beta'\] are not parameters of the 'rate' parametrization",
                id="exponential-rate-beta",
            ),
        ],
    )
    def test_rejects_unknown_parameters(
        self, make_component: Callable[[], ParametricFamilyDistribution], params: set[str], message: str
    ) -> None:
        """Reject unknown parameter names with the pair-specific message."""
        component = make_component()
        state = _analytical_state(component)

        with pytest.raises(ValueError, match=message):
            q_function_strategy(component, state, q_block(params=params), ScipyNelderMead())

    def test_updates_nothing_for_negligible_responsibility(
        self, analytical_component: ParametricFamilyDistribution
    ) -> None:
        """Return the unchanged component id and no parameters for negligible responsibility."""
        state = _analytical_state(analytical_component)
        assert state.H is not None
        state.H.fill(1e-10)

        assert q_function_strategy(analytical_component, state, q_block(), ScipyNelderMead()) == (0, {})

    def test_view_with_fixed_mean_updates_only_the_standard_deviation(self, normal_state: PipelineState) -> None:
        """View with fixed mean updates only the standard deviation."""
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

        _, new_params = q_function_strategy(fixed_mu, normal_state, q_block(), optimizer=ScipyNelderMead())

        assert set(new_params) == {"sigma"}

    def test_view_with_fixed_standard_deviation_updates_only_the_mean(self, normal_state: PipelineState) -> None:
        """View with fixed standard deviation updates only the mean."""
        fixed_sigma = family(FamilyName.NORMAL).view(sigma=2.0)(mu=0.0)

        _, new_params = q_function_strategy(fixed_sigma, normal_state, q_block(), optimizer=ScipyNelderMead())

        assert set(new_params) == {"mu"}

    def test_rejects_a_fixed_parameter(self, normal_state: PipelineState) -> None:
        """Rejects a fixed parameter."""
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

        with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
            q_function_strategy(fixed_mu, normal_state, q_block(params={"mu", "sigma"}), optimizer=ScipyNelderMead())


class TestTotalResponsibility:
    """Boundary behavior with an explicitly requested threshold."""

    @pytest.mark.parametrize(
        "make_component", [lambda: exponential(2.5), lambda: normal(0.0, 2.0)], ids=["exp", "normal"]
    )
    def test_at_the_tolerance_nothing_is_updated(
        self, coarse_tolerance: None, make_component: Callable[[], ParametricFamilyDistribution]
    ) -> None:
        """At the tolerance nothing is updated."""
        assert _run_threshold(make_component(), [1.0], [[TOLERANCE]]) == {}

    @pytest.mark.parametrize(
        "make_component", [lambda: exponential(2.5), lambda: normal(0.0, 2.0)], ids=["exp", "normal"]
    )
    def test_above_the_tolerance_the_parameters_are_updated(
        self, coarse_tolerance: None, make_component: Callable[[], ParametricFamilyDistribution]
    ) -> None:
        """Above the tolerance the parameters are updated."""
        assert _run_threshold(make_component(), [1.0], [[0.5]]) != {}


class TestExponentialMean:
    """Boundary behavior with an explicitly requested threshold."""

    def test_at_the_tolerance_the_rate_is_kept(self, coarse_tolerance: None) -> None:
        """At the tolerance the rate is kept."""
        assert _run_threshold(exponential(2.5), [TOLERANCE], [[1.0]]) == {"lambda_": 2.5}

    def test_above_the_tolerance_the_rate_is_updated(self, coarse_tolerance: None) -> None:
        """Above the tolerance the rate is updated."""
        assert _run_threshold(exponential(2.5), [0.5], [[1.0]]) == {"lambda_": 2.0}


class TestNormalVariance:
    """Boundary behavior with an explicitly requested threshold."""

    def test_at_the_tolerance_the_standard_deviation_is_kept(self, coarse_tolerance: None) -> None:
        """At the tolerance the standard deviation is kept."""
        # The mean is 0.5 and the variance is exactly 0.25.
        result = _run_threshold(normal(0.0, 2.0), [0.0, 1.0], [[0.5], [0.5]])

        assert result == {"mu": 0.5, "sigma": 2.0}

    def test_above_the_tolerance_the_standard_deviation_is_updated(self, coarse_tolerance: None) -> None:
        """Above the tolerance the standard deviation is updated."""
        # The mean is 1 and the variance is 1.
        result = _run_threshold(normal(0.0, 2.0), [0.0, 2.0], [[0.5], [0.5]])

        assert result == {"mu": 1.0, "sigma": 1.0}


class TestNumericalOptimization:
    """Numerical optimization with real optimizer implementations."""

    @pytest.mark.parametrize("optimizer_class", [ScipyNelderMead, ScipyPowell], ids=["nelder-mead", "powell"])
    def test_recovers_the_parameters_of_a_sample(self, optimizer_class: type[Optimizer]) -> None:
        """Recovers the parameters of a sample."""
        state = sample_state(true_mu=2.0, true_var=2.25)
        start = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)

        _, new_params = q_function_strategy(start, state, q_block(), optimizer_class())

        assert new_params["mu"] == pytest.approx(np.mean(state.X), abs=1e-3)
        assert new_params["var"] == pytest.approx(np.var(state.X), rel=1e-3)

    @pytest.mark.parametrize("optimizer_class", [ScipyNelderMead, ScipyPowell], ids=["nelder-mead", "powell"])
    def test_survives_steps_outside_the_constraints(self, optimizer_class: type[Optimizer]) -> None:
        """Survives steps outside the constraints."""
        # Without the infinite target such a step would raise ValueError from pysatl-core.
        state = sample_state(true_mu=0.0, true_var=1e-4)
        start = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)

        _, new_params = q_function_strategy(start, state, q_block(params={"var"}), optimizer_class())

        assert new_params["var"] == pytest.approx(np.mean(state.X**2), rel=1e-3)

    def test_agrees_with_the_analytical_solution_for_the_normal_family(self) -> None:
        """Agrees with the analytical solution for the normal family."""
        state = weighted_state()
        normal = family(FamilyName.NORMAL)

        _, analytical = q_function_strategy(normal(mu=0.0, sigma=1.0), state, q_block(), ScipyNelderMead())  # meanStd
        _, numerical = q_function_strategy(
            normal(parametrization_name="meanVar", mu=0.0, var=1.0), state, q_block(), ScipyNelderMead()
        )

        assert numerical["mu"] == pytest.approx(analytical["mu"], rel=1e-3)
        assert numerical["var"] == pytest.approx(analytical["sigma"] ** 2, rel=1e-3)

    def test_agrees_with_the_analytical_solution_for_the_exponential_family(self) -> None:
        """Agrees with the analytical solution for the exponential family."""
        state = weighted_state()
        exponential = family(FamilyName.EXPONENTIAL)

        _, analytical = q_function_strategy(exponential(lambda_=1.0), state, q_block(), ScipyNelderMead())  # rate
        _, numerical = q_function_strategy(
            exponential(parametrization_name="scale", beta=1.0), state, q_block(), ScipyNelderMead()
        )

        assert 1.0 / numerical["beta"] == pytest.approx(analytical["lambda_"], rel=1e-3)
        assert_allclose(numerical["beta"], 1.0 / analytical["lambda_"], rtol=1e-3)
