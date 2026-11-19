"""Tests for prior threshold pruning behavior."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import PipelineState, PriorThresholdPruner

from tests.helpers.families import family


def make_component(index: int) -> ParametricFamilyDistribution:
    """Return an exponential component with a unique rate, so that different indices give different components."""

    return family(FamilyName.EXPONENTIAL)(lambda_=float(index + 1))


# --- Fixtures ---


@pytest.fixture
def dummy_components() -> list[ParametricFamilyDistribution]:
    """Fixture that provides a list of unique components."""

    return [make_component(i) for i in range(3)]


# --- Tests for __init__ ---


def test_initialization_stores_configured_threshold():
    threshold = 0.1
    pruner = PriorThresholdPruner(threshold)
    assert pruner.threshold == threshold


@pytest.mark.parametrize(
    "invalid_threshold",
    [0.0, 1.0, -0.1, 1.1, -100, 100],
    ids=["zero", "one", "negative", "greater_than_one", "large_negative", "large_positive"],
)
def test_init_raises_value_error_for_invalid_threshold(invalid_threshold):
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        PriorThresholdPruner(invalid_threshold)


# --- Tests for the prune method ---


@pytest.mark.parametrize(
    "threshold, initial_weights, expected_n_components, expected_remaining_indices, expected_removed_indices",
    [
        (0.05, [0.4, 0.3, 0.3], 3, [0, 1, 2], []),
        (0.15, [0.8, 0.1, 0.1], 1, [0], [1, 2]),
        (0.2, [0.7, 0.15, 0.15], 1, [0], [1, 2]),
        (0.1, [0.8, 0.15, 0.05], 2, [0, 1], [2]),
        (0.1, [0.8, 0.1, 0.1], 3, [0, 1, 2], []),
    ],
    ids=[
        "no_pruning",
        "prune_one_component",
        "prune_two_components",
        "prune_smallest_weight",
        "weight_equals_threshold",
    ],
)
def test_pruning_removes_subthreshold_components_and_keeps_threshold_equal_weights(
    dummy_components,
    threshold,
    initial_weights,
    expected_n_components,
    expected_remaining_indices,
    expected_removed_indices,
):
    pruner = PriorThresholdPruner(threshold)
    components = [make_component(i) for i in range(len(initial_weights))]
    initial_mixture = MixtureModel(components, weights=initial_weights)
    state = PipelineState(
        X=np.array([], dtype=np.float64), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None
    )

    new_state, removed_components_indices = pruner.prune(state)
    assert removed_components_indices == expected_removed_indices

    new_mixture = new_state.curr_mixture

    assert new_mixture.n_components == expected_n_components
    expected_components = tuple(components[i] for i in expected_remaining_indices)
    assert new_mixture.components == expected_components
    assert np.isclose(np.sum(new_mixture.weights), 1.0)


def test_prune_keeps_first_component_on_equal_weights_all_below_threshold():
    pruner = PriorThresholdPruner(threshold=0.9)
    # Mixture with two components, both below threshold; equal weights resolve to the smallest index
    components = [make_component(0), make_component(1)]
    initial_mixture = MixtureModel(components[:2], weights=[0.5, 0.5])
    state = PipelineState(
        X=np.array([], dtype=np.float64), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None
    )

    new_state, removed = pruner.prune(state)
    assert removed == [1]
    assert new_state.curr_mixture.n_components == 1
    assert new_state.curr_mixture.components == (components[0],)


def test_prune_preserves_other_pipeline_state_attributes(dummy_components):
    pruner = PriorThresholdPruner(threshold=0.5)

    X_data = np.array([1, 2, 3])
    H_data = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    prev_mix = MixtureModel([make_component(10)], weights=[1.0])
    curr_mix = MixtureModel(dummy_components[:2], weights=[0.1, 0.9])
    error_obj = ValueError("test error")

    initial_state = PipelineState(X=X_data, H=H_data, prev_mixture=prev_mix, curr_mixture=curr_mix, error=error_obj)

    initial_X_id = id(initial_state.X)
    initial_H_id = id(initial_state.H)
    initial_prev_mix_id = id(initial_state.prev_mixture)
    initial_error_id = id(initial_state.error)
    initial_curr_mix_id = id(initial_state.curr_mixture)

    new_state, _ = pruner.prune(initial_state)

    assert id(new_state.X) == initial_X_id
    assert id(new_state.H) == initial_H_id
    assert id(new_state.prev_mixture) == initial_prev_mix_id
    assert id(new_state.error) == initial_error_id

    # curr_mixture is replaced with a pruned copy
    assert id(new_state.curr_mixture) != initial_curr_mix_id
    assert new_state.curr_mixture.n_components == 1


def test_prune_does_not_modify_original_mixture_object(dummy_components):
    pruner = PriorThresholdPruner(threshold=0.5)
    initial_mixture = MixtureModel(dummy_components[:2], weights=[0.1, 0.9])
    components_before = initial_mixture.components
    weights_before = initial_mixture.weights.copy()
    log_weights_before = initial_mixture.log_weights.copy()

    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    pruner.prune(state)

    assert initial_mixture.n_components == len(components_before)
    assert np.allclose(initial_mixture.weights, weights_before)
    assert np.array_equal(initial_mixture.log_weights, log_weights_before)
    assert initial_mixture.components == components_before


# --- Tests using Hypothesis ---


@given(
    weights_list=st.lists(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False), min_size=1, max_size=10),
    threshold=st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False),
)
def test_pruning_preserves_valid_mixture_invariants_for_generated_weights(weights_list, threshold):
    weights_sum = sum(weights_list)
    if weights_sum == 0:
        return
    initial_weights = np.array(weights_list) / weights_sum

    num_components = len(initial_weights)
    components = [make_component(i) for i in range(num_components)]

    pruner = PriorThresholdPruner(threshold)
    initial_mixture = MixtureModel(components, weights=initial_weights)
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, _ = pruner.prune(state)
    new_mixture = new_state.curr_mixture

    assert new_mixture.n_components >= 1
    assert new_mixture.n_components <= initial_mixture.n_components

    initial_components_set = set(initial_mixture.components)
    final_components_set = set(new_mixture.components)
    removed_components = initial_components_set - final_components_set

    if len(removed_components) < initial_mixture.n_components - 1:
        for i, comp in enumerate(initial_mixture.components):
            if comp in removed_components:
                assert initial_mixture.weights[i] < threshold

    assert np.isclose(np.sum(new_mixture.weights), 1.0)


@given(data=st.data())
def test_prune_keeps_largest_weight_component_when_all_weights_below_threshold(data):
    weights_list = data.draw(
        st.lists(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False), min_size=2, max_size=10)
    )
    weights_sum = sum(weights_list)
    initial_weights = np.array(weights_list) / weights_sum

    num_components = len(initial_weights)
    components = [make_component(i) for i in range(num_components)]
    initial_mixture = MixtureModel(components, weights=initial_weights)

    mixture_weights = initial_mixture.weights
    w_max = float(mixture_weights.max())
    assume(w_max < 1.0 - 1e-6)
    threshold = data.draw(st.floats(min_value=w_max, max_value=1.0 - 1e-6, exclude_min=True, allow_nan=False))

    pruner = PriorThresholdPruner(threshold)
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, removed = pruner.prune(state)
    new_mixture = new_state.curr_mixture

    expected_survivor = components[int(np.argmax(mixture_weights))]
    assert new_mixture.n_components == 1
    assert new_mixture.components == (expected_survivor,)
    assert len(removed) == num_components - 1
    assert np.isclose(np.sum(new_mixture.weights), 1.0)


def test_prune_returns_original_state_when_no_pruning(dummy_components):
    pruner = PriorThresholdPruner(threshold=0.05)
    initial_mixture = MixtureModel(dummy_components, weights=[0.4, 0.3, 0.3])
    original_state = PipelineState(
        X=np.array([1.0]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None
    )

    new_state, removed = pruner.prune(original_state)

    assert removed == []
    assert new_state is original_state
    assert new_state.curr_mixture is initial_mixture


def test_prune_returns_new_state_and_new_mixture_objects(dummy_components):
    pruner = PriorThresholdPruner(threshold=0.15)
    initial_mixture = MixtureModel(dummy_components, weights=[0.7, 0.2, 0.1])
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, _ = pruner.prune(state)

    assert new_state is not state
    assert new_state.curr_mixture is not initial_mixture


def test_prune_result_has_expected_renormalized_weights(dummy_components):
    pruner = PriorThresholdPruner(threshold=0.15)
    initial_mixture = MixtureModel(dummy_components, weights=[0.7, 0.2, 0.1])
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, removed = pruner.prune(state)

    expected_removed = [2]
    expected_n_components = 2
    expected_weights = np.array([0.7 / 0.9, 0.2 / 0.9])
    assert removed == expected_removed
    new_mixture = new_state.curr_mixture
    assert new_mixture.n_components == expected_n_components
    assert np.allclose(new_mixture.weights, expected_weights)
    assert new_mixture.components[0] is dummy_components[0]
    assert new_mixture.components[1] is dummy_components[1]


def test_prune_all_below_threshold_keeps_largest_weight_component():
    pruner = PriorThresholdPruner(threshold=0.5)
    components = [make_component(i) for i in range(4)]
    initial_mixture = MixtureModel(components, weights=[0.4, 0.3, 0.2, 0.1])
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, removed = pruner.prune(state)

    expected_removed = [1, 2, 3]
    expected_weights = np.array([1.0])
    assert removed == expected_removed
    new_mixture = new_state.curr_mixture
    assert new_mixture.n_components == 1
    assert new_mixture.components == (components[0],)
    assert np.allclose(new_mixture.weights, expected_weights)


def test_prune_all_below_threshold_with_heaviest_first_component():
    pruner = PriorThresholdPruner(threshold=0.9)
    components = [make_component(i) for i in range(3)]
    initial_mixture = MixtureModel(components, weights=[0.5, 0.3, 0.2])
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, removed = pruner.prune(state)

    assert removed == [1, 2]
    new_mixture = new_state.curr_mixture
    assert new_mixture.n_components == 1
    assert new_mixture.components == (components[0],)
    assert np.allclose(new_mixture.weights, [1.0])


def test_prune_all_below_threshold_with_heaviest_middle_component():
    pruner = PriorThresholdPruner(threshold=0.9)
    components = [make_component(i) for i in range(3)]
    initial_mixture = MixtureModel(components, weights=[0.3, 0.4, 0.3])
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state, removed = pruner.prune(state)

    assert removed == [0, 2]
    new_mixture = new_state.curr_mixture
    assert new_mixture.n_components == 1
    assert new_mixture.components == (components[1],)
    assert np.allclose(new_mixture.weights, [1.0])
