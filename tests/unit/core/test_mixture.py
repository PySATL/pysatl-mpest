"""Unit tests for MixtureModel core class."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterator
from copy import copy
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike, NDArray
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import CharacteristicName
from pysatl_mpest.core.mixture import MixtureModel
from pysatl_mpest.typings import FloatArray

from tests.helpers.math_assertions import assert_is_array_type, assert_is_scalar_type
from tests.mocks.distributions import MockContinuousDistribution


def assert_probabilities_sum_to_one(weights: FloatArray, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """
    Asserts that the elements of the probability (or weights) array sum to one.
    If the array is multi-dimensional, the sum is checked along the last axis (axis=-1).
    """

    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, rtol=rtol, atol=atol)


def assert_computational_stability(log_probs: FloatArray) -> None:
    """
    Asserts the computational stability of a log-probabilities (lpdf) tensor.
    Ensures no NaN values are present. For Inf, only -Inf is allowed
    (in case of log(0)), but +Inf is strictly forbidden.
    """

    assert not np.isnan(log_probs).any(), "Log-probabilities array contains NaN"
    assert not np.isposinf(log_probs).any(), "Log-probabilities array contains +Inf"


class MockConstantLpdfContinuousDistribution(MockContinuousDistribution):
    """Mock distribution with constant log-PDF."""

    def __init__(self, log_density: float) -> None:
        super().__init__(param1=log_density)
        self.log_density = log_density

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.full_like(X_arr, self.log_density)
        raise NotImplementedError


class MockInfLpdfContinuousDistribution(MockContinuousDistribution):
    """Mock distribution with negative infinite log-PDF."""

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.full_like(X_arr, -np.inf)
        raise NotImplementedError


@pytest.fixture
def mock_comp() -> ParametricFamilyDistribution:
    return cast(ParametricFamilyDistribution, MockContinuousDistribution())


@pytest.fixture
def mock_comps() -> tuple[ParametricFamilyDistribution, ParametricFamilyDistribution]:
    return (
        cast(ParametricFamilyDistribution, MockContinuousDistribution()),
        cast(ParametricFamilyDistribution, MockContinuousDistribution()),
    )


@pytest.fixture
def param_comps() -> tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution]:
    return (
        cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=1.0)),
        cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=2.0)),
        cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=3.0)),
    )


@pytest.fixture
def mock_finite_mixture() -> Iterator[MagicMock]:
    """Observe construction of core mixtures without constructing one."""
    with patch("pysatl_mpest.core.mixture.finite_mixture") as mock_fm:
        yield mock_fm


class TestConstruction:
    def test_default_weights_are_uniform(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        rtol, atol = 1e-5, 1e-8

        expected_components = 2
        assert mixture.n_components == expected_components
        assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
        assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)

    def test_custom_weights_are_preserved(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.2, 0.8])
        rtol, atol = 1e-5, 1e-8

        expected_components = 2
        assert mixture.n_components == expected_components
        assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
        assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)

    def test_zero_weight_is_preserved(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.0, 1.0])
        rtol, atol = 1e-5, 1e-8

        assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
        assert_allclose(mixture.weights, [0.0, 1.0], rtol=rtol, atol=atol)

    def test_empty_components_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"^List of components cannot be empty$"):
            MixtureModel([])

    def test_invalid_weights_are_rejected(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        with pytest.raises(ValueError, match=r"^Weights must be positive\.$"):
            MixtureModel(mock_comps, weights=[-0.1, 1.1])

        with pytest.raises(ValueError, match=r"^Sum of the weights must be equal 1, but it equal 1\.1\.$"):
            MixtureModel(mock_comps, weights=[0.5, 0.6])

        with pytest.raises(ValueError, match="must be equal to weights number"):
            MixtureModel(mock_comps, weights=[1.0])

    def test_default_log_weights_are_normalized(self) -> None:
        components = [cast(ParametricFamilyDistribution, MockContinuousDistribution()) for _ in range(3)]

        mixture = MixtureModel(components)

        assert_allclose(mixture.log_weights, np.log(np.full(3, 1.0 / 3.0)))

    def test_initial_log_weights_are_normalized(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.3, 0.700001])

        assert np.sum(np.exp(mixture.log_weights)) == pytest.approx(1.0, rel=0.0, abs=1e-12)


class TestLogWeights:
    def test_logarithms_of_weights_are_returned(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.3, 0.7])
        expected = np.log(np.array([0.3, 0.7], dtype=np.float64) + np.finfo(np.float64).tiny)
        assert_allclose(mixture.log_weights, expected)

    @pytest.mark.parametrize("offset", [-1e100, 1e100], ids=["negative", "positive"])
    def test_normalization_handles_large_common_offsets(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...], offset: float
    ) -> None:
        mixture = MixtureModel(mock_comps)
        result = mixture.with_log_weights([offset, offset])

        assert result.log_weights == pytest.approx(np.log([0.5, 0.5]))
        assert np.sum(np.exp(result.log_weights)) == pytest.approx(1.0, abs=1e-12)

    def test_with_log_weights_copies_input_array(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        raw = np.array([0.0, 0.0])
        result = mixture.with_log_weights(raw)
        before = result.log_weights.copy()

        raw[0] = 999.0

        assert np.array_equal(result.log_weights, before)

    def test_getter_is_read_only(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        before = mixture.log_weights.copy()

        with pytest.raises(ValueError, match="read-only"):
            mixture.log_weights[0] = 0.0

        assert np.array_equal(mixture.log_weights, before)

    def test_with_log_weights_accepts_zero_weight(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        result = mixture.with_log_weights([-np.inf, 0.0])

        assert np.array_equal(result.log_weights, [-np.inf, 0.0])
        assert np.array_equal(result.weights, [0.0, 1.0])


class TestComponents:
    def test_with_added_component_keeps_the_proportions_of_several_components(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])
        result = mixture.with_added_component(comp3, weight=0.5)

        assert_allclose(result.weights, [0.2, 0.3, 0.5])
        assert_allclose(result.log_weights, np.log([0.2, 0.3, 0.5]))

    def test_with_added_component_with_prior_unnormalized_weights(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...]
    ) -> None:
        mixture = MixtureModel(mock_comps).with_log_weights(np.log([2.0, 6.0]))

        result = mixture.with_added_component(mock_comps[0], weight=0.5)

        assert result.weights == pytest.approx([0.125, 0.375, 0.5])
        assert np.sum(np.exp(result.log_weights)) == pytest.approx(1.0, abs=1e-12)
        assert not result.log_weights.flags.writeable

    def test_derived_models_do_not_modify_the_input_array(
        self,
        mock_comps: tuple[ParametricFamilyDistribution, ...],
    ) -> None:
        mixture = MixtureModel(mock_comps)
        raw = np.log([0.5, 0.5])
        before = raw.copy()

        mixture.with_log_weights(raw).with_added_component(mock_comps[0], weight=0.5)

        assert np.array_equal(raw, before)

    def test_indexing_and_iteration_return_components(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, _ = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

        assert mixture1[0] is comp1
        assert mixture1[1] is comp2

        comps = list(mixture1)
        assert comps == [comp1, comp2]

    def test_without_component_keeps_log_weights_read_only(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...]
    ) -> None:
        mixture = MixtureModel(mock_comps).with_log_weights([-np.inf, 0.0])

        result = mixture.without_component(0)

        assert np.array_equal(result.log_weights, [0.0])
        assert not result.log_weights.flags.writeable


class TestImmutableMethods:
    def test_with_component_returns_new_model_and_preserves_original(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])
        original_log_weights = mixture.log_weights.copy()
        original_hash = hash(mixture)

        result = mixture.with_component(1, comp3)

        expected_components = 2
        assert result is not mixture
        assert result.n_components == expected_components
        assert result.components[0] is comp1
        assert result.components[1] is comp3
        assert_allclose(result.log_weights, original_log_weights)
        assert_allclose(result.weights, [0.4, 0.6])
        assert not result.log_weights.flags.writeable

        assert mixture.n_components == expected_components
        assert mixture.components[0] is comp1
        assert mixture.components[1] is comp2
        assert np.array_equal(mixture.log_weights, original_log_weights)
        assert_allclose(mixture.weights, [0.4, 0.6])
        assert hash(mixture) == original_hash

    @pytest.mark.parametrize("index", [-1, 2, 99])
    def test_with_component_rejects_out_of_range_index(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...], index: int
    ) -> None:
        mixture = MixtureModel(mock_comps)
        comp = cast(ParametricFamilyDistribution, MockContinuousDistribution())

        with pytest.raises(IndexError, match=r"^Index of component must be between 0 and 1$"):
            mixture.with_component(index, comp)

    def test_with_components_replaces_all_components_and_preserves_log_weights(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2], weights=[0.3, 0.7])
        original_log_weights = mixture.log_weights.copy()

        result = mixture.with_components([comp3, comp1])

        expected_components = 2
        assert result is not mixture
        assert result.n_components == expected_components
        assert result.components[0] is comp3
        assert result.components[1] is comp1
        assert_allclose(result.log_weights, original_log_weights)
        assert_allclose(result.weights, [0.3, 0.7])
        assert not result.log_weights.flags.writeable

        assert mixture.components[0] is comp1
        assert mixture.components[1] is comp2
        assert np.array_equal(mixture.log_weights, original_log_weights)

    def test_with_components_rejects_wrong_length(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)

        with pytest.raises(ValueError, match=r"^Expected 2 components, got 1\.$"):
            mixture.with_components([mock_comps[0]])

    def test_with_log_weights_normalizes_input_and_preserves_original(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...]
    ) -> None:
        mixture = MixtureModel(mock_comps)
        raw = np.log([2.0, 6.0])
        raw_before = raw.copy()
        original_log_weights = mixture.log_weights.copy()

        result = mixture.with_log_weights(raw)

        assert result is not mixture
        assert_allclose(result.log_weights, np.log([0.25, 0.75]))
        assert_allclose(result.weights, [0.25, 0.75])
        assert not result.log_weights.flags.writeable

        assert np.array_equal(raw, raw_before)
        assert np.array_equal(mixture.log_weights, original_log_weights)
        assert_allclose(mixture.weights, [0.5, 0.5])

    def test_with_log_weights_rejects_wrong_length(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)

        with pytest.raises(ValueError, match="length of the new log-weights vector"):
            mixture.with_log_weights(np.array([0.0]))

    @pytest.mark.parametrize(
        "log_weights",
        [[-np.inf, -np.inf], [np.nan, 0.0], [np.inf, 0.0]],
        ids=["all-negative-inf", "nan", "positive-inf"],
    )
    def test_with_log_weights_rejects_values_that_cannot_be_normalized(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...], log_weights: list[float]
    ) -> None:
        mixture = MixtureModel(mock_comps)
        before = mixture.log_weights.copy()

        with pytest.raises(ValueError, match="normalizable"):
            mixture.with_log_weights(log_weights)

        assert np.array_equal(mixture.log_weights, before)

    def test_with_added_component_preserves_proportions(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2], weights=[0.25, 0.75])
        original_weights = mixture.weights.copy()
        original_log_weights = mixture.log_weights.copy()

        result = mixture.with_added_component(comp3, weight=0.2)

        expected_components = 3
        assert result is not mixture
        assert result.n_components == expected_components
        assert result.components[0] is comp1
        assert result.components[1] is comp2
        assert result.components[2] is comp3
        assert_allclose(result.weights, [0.2, 0.6, 0.2])
        assert_allclose(result.log_weights, np.log([0.2, 0.6, 0.2]))
        assert not result.log_weights.flags.writeable

        assert mixture.n_components == expected_components - 1
        assert_allclose(mixture.weights, original_weights)
        assert np.array_equal(mixture.log_weights, original_log_weights)

    def test_with_added_component_rejects_invalid_weight(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
        original_n_components = mixture.n_components

        message = r"^The weight of the new component must be in the range \(0, 1\)\.$"
        for weight in (-0.1, 0.0, 1.0, 1.1):
            with pytest.raises(ValueError, match=message):
                mixture.with_added_component(comp3, weight=weight)

        assert mixture.n_components == original_n_components

    def test_without_component_renormalizes_weights(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2, comp3], weights=[0.2, 0.3, 0.5])

        result = mixture.without_component(1)

        expected_components = 2
        assert result is not mixture
        assert result.n_components == expected_components
        assert result.components[0] is comp1
        assert result.components[1] is comp3
        assert_allclose(result.weights, [0.2 / 0.7, 0.5 / 0.7])
        assert_allclose(result.log_weights, np.log([0.2 / 0.7, 0.5 / 0.7]))
        assert not result.log_weights.flags.writeable

        assert mixture.n_components == expected_components + 1
        assert_allclose(mixture.weights, [0.2, 0.3, 0.5])

    @pytest.mark.parametrize("index", [-1, 3, 99])
    def test_without_component_rejects_out_of_range_index(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
        index: int,
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2, comp3], weights=[0.2, 0.3, 0.5])
        original_n_components = mixture.n_components

        with pytest.raises(IndexError, match=rf"^Index {index} out of range \[0, 2\]$"):
            mixture.without_component(index)

        assert mixture.n_components == original_n_components

    def test_without_component_rejects_removal_of_last_component(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture = MixtureModel([mock_comp])

        with pytest.raises(ValueError, match=r"^The last component cannot be removed from the model\.$"):
            mixture.without_component(0)

    def test_without_component_rejects_removal_of_only_positive_weight(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...]
    ) -> None:
        mixture = MixtureModel(mock_comps).with_log_weights([-np.inf, 0.0])
        before = mixture.log_weights.copy()

        with pytest.raises(ValueError, match="positive weight"):
            mixture.without_component(1)

        expected_components = 2
        assert mixture.n_components == expected_components
        assert np.array_equal(mixture.log_weights, before)
        assert_allclose(mixture.weights, [0.0, 1.0])

    def test_derived_models_do_not_share_caches(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])
        _ = mixture.weights
        _ = hash(mixture)

        result = mixture.with_component(0, comp3).with_log_weights(np.log([0.5, 0.5]))

        expected = MixtureModel([comp3, comp2], weights=[0.5, 0.5])
        assert result == expected

        assert mixture == MixtureModel([comp1, comp2], weights=[0.4, 0.6])
        assert_allclose(mixture.weights, [0.4, 0.6])
        assert hash(mixture) == hash(MixtureModel([comp1, comp2], weights=[0.4, 0.6]))


class TestImmutability:
    def test_setitem_is_not_supported(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        replacement = cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=99.0))

        with pytest.raises(TypeError):
            mixture[0] = replacement

        assert mixture.components == mock_comps

    def test_log_weights_cannot_be_assigned(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        before = mixture.log_weights.copy()

        with pytest.raises(AttributeError, match="log_weights"):
            mixture.log_weights = np.array([0.0, 0.0])

        assert np.array_equal(mixture.log_weights, before)

    def test_mutators_are_removed(self) -> None:
        assert not hasattr(MixtureModel, "add_component")
        assert not hasattr(MixtureModel, "remove_component")
        assert not hasattr(MixtureModel, "__setitem__")

    def test_components_is_a_tuple(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)

        assert isinstance(mixture.components, tuple)

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(lambda mixture, extra: mixture.with_component(0, extra), id="with_component"),
            pytest.param(lambda mixture, extra: mixture.with_components([extra, extra]), id="with_components"),
            pytest.param(lambda mixture, extra: mixture.with_log_weights(np.log([0.9, 0.1])), id="with_log_weights"),
            pytest.param(
                lambda mixture, extra: mixture.with_added_component(extra, weight=0.5), id="with_added_component"
            ),
            pytest.param(lambda mixture, extra: mixture.without_component(0), id="without_component"),
        ],
    )
    def test_derivation_keeps_the_original_usable_as_dict_key(
        self,
        mock_comps: tuple[ParametricFamilyDistribution, ...],
        derive: Callable[[MixtureModel, ParametricFamilyDistribution], MixtureModel],
    ) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.5, 0.5])
        registry = {mixture: "start"}
        extra = cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=99.0))

        derived = derive(mixture, extra)

        assert registry[mixture] == "start"
        assert derived not in registry


class TestDensity:
    def test_pdf_is_exponential_of_lpdf(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.4, 0.6])
        rtol, atol = 1e-5, 1e-8

        x_scalar = 1.0
        lpdf_scalar = mixture.lpdf(x_scalar)
        pdf_scalar = mixture.pdf(x_scalar)
        assert_allclose(pdf_scalar, np.exp(lpdf_scalar), rtol=rtol, atol=atol)

        x_array = np.array([1.0, 2.0, 3.0])
        lpdf_array = mixture.lpdf(x_array)
        pdf_array = mixture.pdf(x_array)

        assert lpdf_array.shape == (3,)
        assert pdf_array.shape == (3,)
        assert_allclose(pdf_array, np.exp(lpdf_array), rtol=rtol, atol=atol)

    def test_returns_stable_log_densities(self) -> None:
        comp1 = cast(ParametricFamilyDistribution, MockInfLpdfContinuousDistribution())
        comp2 = cast(ParametricFamilyDistribution, MockContinuousDistribution())
        mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

        X = np.array([-1.0, 0.0, 1.0])
        lpdf_vals = mixture.lpdf(X)
        assert_computational_stability(cast(np.ndarray, lpdf_vals))

    def test_lpdf_does_not_use_core_mixture(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...], mock_finite_mixture: MagicMock
    ) -> None:
        mixture = MixtureModel(mock_comps)

        mixture.lpdf(np.array([1.0, 2.0]))

        mock_finite_mixture.assert_not_called()

    def test_lpdf_is_weighted_logsumexp(self) -> None:
        comps = (
            cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-1.0)),
            cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-3.0)),
        )
        mixture = MixtureModel(comps, weights=[0.25, 0.75])
        expected = np.log(0.25 * np.exp(-1.0) + 0.75 * np.exp(-3.0))
        rtol, atol = 1e-12, 1e-12

        assert_allclose(mixture.lpdf(np.array([0.0, 1.0])), [expected, expected], rtol=rtol, atol=atol)
        assert_allclose(mixture.lpdf(0.5), expected, rtol=rtol, atol=atol)

    @pytest.mark.filterwarnings("error")
    def test_lpdf_all_components_neg_inf(self) -> None:
        comps = (
            cast(ParametricFamilyDistribution, MockInfLpdfContinuousDistribution()),
            cast(ParametricFamilyDistribution, MockInfLpdfContinuousDistribution()),
        )
        mixture = MixtureModel(comps)

        assert np.all(np.isneginf(mixture.lpdf(np.array([-1.0, 0.0, 1.0]))))

    def test_component_lpdf_matrix_shape(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        n_components = 2

        assert mixture.component_lpdf_matrix(np.zeros(5)).shape == (5, n_components)
        assert mixture.component_lpdf_matrix(0.5).shape == (n_components,)
        assert mixture.component_lpdf_matrix(np.zeros((3, 4))).shape == (3, 4, n_components)

    def test_component_lpdf_matrix_columns_are_contiguous(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...]
    ) -> None:
        # Reducing the short component axis (logsumexp(axis=-1) in E-step) is several times slower
        # on row-major (n, k) matrix than on column-contiguous.
        mixture = MixtureModel(mock_comps)

        matrix = mixture.component_lpdf_matrix(np.zeros(5))

        assert all(matrix[:, j].flags.c_contiguous for j in range(matrix.shape[1]))

    def test_component_lpdf_matrix_values_exclude_weights(self) -> None:
        comps = (
            cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-1.0)),
            cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-3.0)),
        )
        mixture = MixtureModel(comps, weights=[0.25, 0.75])

        matrix = mixture.component_lpdf_matrix(np.array([0.0, 1.0, 2.0]))

        assert_allclose(matrix, np.tile([-1.0, -3.0], (3, 1)))

    def test_loglikelihood_is_sum_of_lpdf(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps, weights=[0.5, 0.5])
        X = np.array([1.0, 2.5, 3.1])

        ll = mixture.loglikelihood(X)
        expected_ll = float(np.sum(mixture.lpdf(X)))
        rtol, atol = 1e-5, 1e-8
        assert_allclose(ll, expected_ll, rtol=rtol, atol=atol)

    def test_pdf_returns_float64_scalar_for_a_scalar(self) -> None:
        mixture = MixtureModel([cast(ParametricFamilyDistribution, MockContinuousDistribution())])

        assert_is_scalar_type(mixture.pdf(1.0))
        assert_is_array_type(mixture.pdf(np.array([1.0, 2.0])), (2,))

    @pytest.mark.parametrize("data", [[1, 2, 3], np.array([1, 2, 3], dtype=np.int32)], ids=["list", "int32"])
    def test_pdf_returns_float64_for_integer_input(self, data: ArrayLike) -> None:
        mixture = MixtureModel([cast(ParametricFamilyDistribution, MockContinuousDistribution())])

        assert_is_array_type(mixture.pdf(data), (3,))

    def test_pdf_does_not_construct_core_mixture(
        self, mock_comps: tuple[ParametricFamilyDistribution, ...], mock_finite_mixture: MagicMock
    ) -> None:
        mixture = MixtureModel(mock_comps)

        mixture.pdf([0.0, 1.0])

        mock_finite_mixture.assert_not_called()


class TestGenerate:
    def test_output_shape_matches_size(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture = MixtureModel([mock_comp])

        sample_scalar = mixture.generate(size=None)
        assert np.isscalar(sample_scalar)

        sample_1d = mixture.generate(size=5)
        assert cast(np.ndarray, sample_1d).shape == (5,)

        sample_2d = mixture.generate(size=(3, 2))
        assert cast(np.ndarray, sample_2d).shape == (3, 2)

    def test_is_reproducible_with_seed(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)
        seed, other_seed, size = 7, 8, 20

        first = cast(np.ndarray, mixture.generate(size=size, random_state=seed))
        second = cast(np.ndarray, mixture.generate(size=size, random_state=seed))
        other = cast(np.ndarray, mixture.generate(size=size, random_state=other_seed))

        assert_allclose(first, second)
        assert not np.allclose(first, other)

    def test_zero_size_produces_empty_arrays(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture = MixtureModel([mock_comp])

        res_int = mixture.generate(size=0)
        assert cast(np.ndarray, res_int).shape == (0,)

        res_tuple = mixture.generate(size=(0, 2))
        assert cast(np.ndarray, res_tuple).shape == (0, 2)

    @pytest.mark.parametrize(
        "size",
        [-1, (-2, 3), (-2, -3), (0, -2)],
        ids=["negative-int", "negative-first", "both-negative", "zero-and-negative"],
    )
    def test_rejects_a_negative_size(
        self, mock_comp: ParametricFamilyDistribution, size: int | tuple[int, ...]
    ) -> None:
        mixture = MixtureModel([mock_comp])

        with pytest.raises(ValueError, match="size must not contain negative dimensions"):
            mixture.generate(size=size, random_state=42)

    def test_zero_tuple_preserves_shape(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture = MixtureModel([mock_comp])

        assert_is_array_type(mixture.generate(size=(0, 5), random_state=42), (0, 5))

    def test_accepts_numpy_integer_size(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture = MixtureModel([mock_comp])

        assert_is_array_type(mixture.generate(size=np.int64(5), random_state=42), (5,))


class TestEquality:
    def test_unrelated_type_is_not_equal(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture = MixtureModel([mock_comp])
        assert mixture != "a string"

    def test_different_component_counts_are_not_equal(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, _ = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
        mixture2 = MixtureModel([comp1])
        assert mixture1 != mixture2

    def test_zero_weight_extra_component_prevents_equality(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        # Zero-weight extra components prevent equality: component count must match.
        comp1, comp2, _ = param_comps
        with_extra = MixtureModel([comp1, comp2], weights=[1.0, 0.0])
        without_extra = MixtureModel([comp1])
        assert with_extra != without_extra
        assert without_extra != with_extra

    def test_component_order_does_not_affect_equality(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, _ = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])
        mixture2 = MixtureModel([comp2, comp1], weights=[0.8, 0.2])
        assert mixture1 == mixture2
        assert hash(mixture1) == hash(mixture2)

    def test_equal_components_with_swapped_weights_are_equal(self, mock_comp: ParametricFamilyDistribution) -> None:
        other = cast(ParametricFamilyDistribution, MockContinuousDistribution())
        mixture1 = MixtureModel([mock_comp, other], weights=[0.2, 0.8])
        mixture2 = MixtureModel([mock_comp, other], weights=[0.8, 0.2])
        assert mixture1 == mixture2
        assert hash(mixture1) == hash(mixture2)

    def test_different_components_are_not_equal(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
        mixture2 = MixtureModel([comp1, comp3], weights=[0.5, 0.5])
        assert mixture1 != mixture2

    def test_different_weights_are_not_equal(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, _ = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
        mixture2 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])
        assert mixture1 != mixture2

    def test_repeated_comparison_remains_equal(self, mock_comp: ParametricFamilyDistribution) -> None:
        mixture1 = MixtureModel([mock_comp])
        mixture2 = MixtureModel([mock_comp])
        assert mixture1 == mixture2
        assert mixture1 == mixture2

    def test_hash_agrees_with_eq_for_weights_within_tolerance(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, _ = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.3, 0.7])
        mixture2 = MixtureModel([comp1, comp2], weights=[0.3, 0.7])
        mixture2 = mixture1.with_log_weights(mixture1.log_weights + np.array([1e-9, 0.0]))

        assert mixture1 == mixture2
        assert hash(mixture1) == hash(mixture2)

    def test_hash_does_not_depend_on_weights(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        # Hash ignores weights: they are compared with tolerance, so cannot be hashed consistently.
        comp1, comp2, _ = param_comps
        mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
        mixture2 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])

        assert mixture1 != mixture2
        assert hash(mixture1) == hash(mixture2)

    def test_hash_differs_for_different_components(
        self,
        param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
    ) -> None:
        comp1, comp2, comp3 = param_comps

        assert hash(MixtureModel([comp1, comp2])) != hash(MixtureModel([comp1, comp3]))


class TestCopy:
    def test_copy_returns_the_same_object(self, mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
        mixture = MixtureModel(mock_comps)

        assert copy(mixture) is mixture
