"""Integration tests for mixture density, equality, sampling, and core export using real families."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.transformations.operations.distributions.mixture import FiniteMixtureDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from scipy import stats

from tests.helpers.families import family
from tests.helpers.math_assertions import assert_is_array_type, assert_is_scalar_type

RTOL, ATOL = 1e-10, 1e-12

Component = tuple[ParametricFamilyDistribution, "stats.rv_continuous"]


def _normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def _normal_fixed_mu(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL).view(mu=mu)(sigma=sigma)


def _reference_normal(mu: float, sigma: float) -> Component:
    return _normal(mu, sigma), stats.norm(loc=mu, scale=sigma)


def _exponential(lambda_: float) -> ParametricFamilyDistribution:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


def _reference_exponential(lambda_: float) -> Component:
    return _exponential(lambda_), stats.expon(scale=1.0 / lambda_)


def _reference_uniform(lower: float, upper: float) -> Component:
    return (
        family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=lower, upper_bound=upper),
        stats.uniform(loc=lower, scale=upper - lower),
    )


class Case:
    """A mixture together with scipy reference densities for its components."""

    def __init__(self, components: Sequence[Component], weights: Sequence[float]) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.references = [reference for _, reference in components]
        self.mixture = MixtureModel([component for component, _ in components], weights=self.weights)

    def reference_lpdf(self, x: np.ndarray) -> np.ndarray:
        """Log-density of the mixture computed with scipy and ``logaddexp``."""

        terms = [np.log(w) + ref.logpdf(x) for w, ref in zip(self.weights, self.references)]
        return np.logaddexp.reduce(terms, axis=0)

    def reference_pdf(self, x: np.ndarray) -> np.ndarray:
        """Density of the mixture computed with scipy."""

        return sum(w * ref.pdf(x) for w, ref in zip(self.weights, self.references))


CASE_FACTORIES: dict[str, Callable[[], Case]] = {
    "normal+normal": lambda: Case([_reference_normal(0.0, 1.0), _reference_normal(3.0, 2.0)], [0.4, 0.6]),
    "normal+exponential": lambda: Case([_reference_normal(0.0, 1.0), _reference_exponential(1.5)], [0.3, 0.7]),
    "normal+uniform": lambda: Case([_reference_normal(2.0, 1.0), _reference_uniform(0.0, 1.0)], [0.5, 0.5]),
}

cases = pytest.mark.parametrize("make_case", CASE_FACTORIES.values(), ids=CASE_FACTORIES.keys())

# Fixed grid: covers the exponential support boundary and the uniform support boundaries.
X_GRID = np.linspace(-3.0, 6.0, 25)


class TestDensity:
    @cases
    def test_pdf_matches_reference(self, make_case: Callable[[], Case]) -> None:
        case = make_case()
        assert_allclose(case.mixture.pdf(X_GRID), case.reference_pdf(X_GRID), rtol=RTOL, atol=ATOL)

    @cases
    def test_lpdf_matches_reference(self, make_case: Callable[[], Case]) -> None:
        case = make_case()
        assert_allclose(case.mixture.lpdf(X_GRID), case.reference_lpdf(X_GRID), rtol=RTOL, atol=ATOL)

    @cases
    def test_pdf_is_exp_of_lpdf(self, make_case: Callable[[], Case]) -> None:
        case = make_case()
        assert_allclose(case.mixture.pdf(X_GRID), np.exp(case.mixture.lpdf(X_GRID)), rtol=RTOL, atol=ATOL)

    def test_lpdf_far_tail_is_finite(self) -> None:
        case = CASE_FACTORIES["normal+normal"]()
        x = np.array([-100.0, 100.0])

        assert np.all(case.mixture.pdf(x) == 0.0)  # documents why lpdf cannot be computed as log(pdf)
        lpdf = case.mixture.lpdf(x)
        assert np.all(np.isfinite(lpdf))
        assert_allclose(lpdf, case.reference_lpdf(x), rtol=RTOL, atol=ATOL)

    @pytest.mark.filterwarnings("error")
    def test_lpdf_outside_every_support_is_neg_inf(self) -> None:
        case = Case([_reference_exponential(1.0), _reference_uniform(0.0, 1.0)], [0.5, 0.5])
        lpdf = case.mixture.lpdf(np.array([-1.0, -0.5]))

        assert np.all(np.isneginf(lpdf))

    @cases
    def test_lpdf_return_types(self, make_case: Callable[[], Case]) -> None:
        mixture = make_case().mixture

        assert_is_scalar_type(mixture.lpdf(0.5))
        assert_is_array_type(mixture.lpdf(X_GRID), X_GRID.shape)

    @cases
    def test_loglikelihood_is_sum_of_lpdf(self, make_case: Callable[[], Case]) -> None:
        case = make_case()
        x = np.linspace(0.1, 5.0, 40)
        assert_allclose(case.mixture.loglikelihood(x), np.sum(case.reference_lpdf(x)), rtol=RTOL, atol=ATOL)

    def test_pdf_matches_exp_lpdf_exactly(self) -> None:
        exponential = family(FamilyName.EXPONENTIAL)
        mixture = MixtureModel([exponential(lambda_=1.0), exponential(lambda_=3.0)])
        mixture = mixture.with_log_weights(np.log([2.0, 6.0]))
        data = np.array([-1.0, 0.0, 0.5, 1.0, 800.0])

        density = mixture.pdf(data)

        assert np.array_equal(density, np.exp(mixture.lpdf(data)))
        assert density[0] == density[-1] == 0.0
        assert np.isneginf(mixture.lpdf(data[0]))
        assert np.isfinite(mixture.lpdf(data[-1]))
        assert isinstance(mixture.pdf(0.5), np.float64)
        assert mixture.pdf(0.5) == np.exp(mixture.lpdf(0.5))
        on_support = data[1:-1]
        assert mixture.loglikelihood(on_support) == pytest.approx(np.sum(np.log(mixture.pdf(on_support))))


class TestEquality:
    def test_independently_built_mixtures_are_equal(self) -> None:
        first = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.3, 0.7])
        second = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.3, 0.7])

        assert first == second
        assert hash(first) == hash(second)

    def test_mixtures_with_view_components_are_equal(self) -> None:
        first = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])
        second = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])

        assert first == second
        assert hash(first) == hash(second)

    def test_view_component_equals_full_component_with_same_values(self) -> None:
        with_view = MixtureModel([_normal_fixed_mu(1.0, 2.0)])
        full = MixtureModel([_normal(1.0, 2.0)])

        assert with_view == full
        assert hash(with_view) == hash(full)

    def test_mixtures_are_equal_when_component_order_differs(self) -> None:
        first = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.3, 0.7])
        second = MixtureModel([_exponential(2.0), _normal(0.0, 1.0)], weights=[0.7, 0.3])

        assert first == second
        assert hash(first) == hash(second)

    def test_mixtures_with_different_fixed_value_differ(self) -> None:
        first = MixtureModel([_normal_fixed_mu(1.0, 2.0)])
        second = MixtureModel([_normal_fixed_mu(2.0, 2.0)])

        assert first != second

    def test_mixtures_with_different_free_value_differ(self) -> None:
        first = MixtureModel([_normal_fixed_mu(1.0, 2.0)])
        second = MixtureModel([_normal_fixed_mu(1.0, 3.0)])

        assert first != second

    def test_zero_weight_component_does_not_make_mixtures_equal(self) -> None:
        component, extra = _normal(0.0, 1.0), _normal(5.0, 1.0)

        assert MixtureModel([component, extra], weights=[1.0, 0.0]) != MixtureModel([component])

    def test_reconstructed_mixture_with_arbitrary_weights_is_equal_and_hashes_equal(self) -> None:
        # Log and softmax re-projection introduces numerical variation around 1e-16.
        components = [_normal(0.0, 1.0), _normal_fixed_mu(3.0, 2.0), _exponential(2.0)]
        rng = np.random.default_rng(0)

        for weights in rng.dirichlet(np.ones(3), size=500):
            mixture = MixtureModel(components, weights=weights)
            duplicate = MixtureModel(mixture.components, mixture.weights)

            assert duplicate == mixture
            assert hash(duplicate) == hash(mixture)

    def test_mixture_is_usable_as_dict_key(self) -> None:
        mixture = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])
        twin = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])

        assert {mixture: "fitted"}[twin] == "fitted"


class TestGenerate:
    @pytest.mark.parametrize(
        "components",
        [
            pytest.param([_normal(0.0, 1.0), _exponential(2.0)], id="plain"),
            pytest.param([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], id="view"),
        ],
    )
    def test_is_reproducible_with_seed(self, components: list[ParametricFamilyDistribution]) -> None:
        mixture = MixtureModel(components, weights=[0.5, 0.5])
        seed, other_seed, size = 42, 43, 200

        first = mixture.generate(size=size, random_state=seed)
        second = mixture.generate(size=size, random_state=seed)
        from_generator = mixture.generate(size=size, random_state=np.random.default_rng(seed))
        other = mixture.generate(size=size, random_state=other_seed)

        assert_allclose(first, second)
        assert_allclose(first, from_generator)
        assert not np.allclose(first, other)

    def test_without_size_returns_single_draw(self) -> None:
        mixture = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.5, 0.5])

        for seed in range(5):
            assert mixture.generate(random_state=seed) == mixture.generate(size=1, random_state=seed)[0]

    def test_preserves_requested_shapes(self) -> None:
        mixture = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.5, 0.5])
        seed = 0

        assert_is_scalar_type(mixture.generate(random_state=seed))
        assert_is_array_type(mixture.generate(size=7, random_state=seed), (7,))
        assert_is_array_type(mixture.generate(size=(3, 4), random_state=seed), (3, 4))

    def test_samples_follow_component_weights(self) -> None:
        mixture = MixtureModel([_normal(0.0, 1.0), _normal_fixed_mu(20.0, 1.0)], weights=[0.5, 0.5])
        size, boundary, tolerance = 20_000, 10.0, 0.02

        sample = np.asarray(mixture.generate(size=size, random_state=2026))
        left, right = sample[sample < boundary], sample[sample >= boundary]

        assert abs(len(left) / size - 0.5) < tolerance
        assert_allclose(np.mean(left), 0.0, atol=0.05)
        assert_allclose(np.mean(right), 20.0, atol=0.05)


class TestCoreExport:
    def test_to_core_mixture_exports_valid_distribution(self) -> None:
        exponential = family(FamilyName.EXPONENTIAL)
        mixture = MixtureModel([exponential(lambda_=1.0), exponential(lambda_=3.0)])
        mixture = mixture.with_log_weights(np.log([2.0, 6.0]))

        exported = mixture.to_core_mixture()

        assert isinstance(exported, FiniteMixtureDistribution)
        assert exported.weights == pytest.approx([0.25, 0.75])
