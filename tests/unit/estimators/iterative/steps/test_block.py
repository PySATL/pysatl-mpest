"""
Integration tests for ``OptimizationBlock.params_to_update`` on real pysatl-core components.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock

from tests.helpers.families import family


def make_block(params: set[str] | None, component_id: int = 3) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.QFUNCTION)


@pytest.fixture
def normal() -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=0.0, sigma=1.0)


@pytest.fixture
def normal_fixed_mu() -> ParametricFamilyDistribution:
    """``mu`` is fixed through a view, ``sigma`` is free."""
    return family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)


def test_none_means_all_free_parameters(normal: ParametricFamilyDistribution) -> None:
    assert make_block(None).params_to_update(normal) == ["mu", "sigma"]


def test_none_excludes_fixed_parameters(normal_fixed_mu: ParametricFamilyDistribution) -> None:
    assert make_block(None).params_to_update(normal_fixed_mu) == ["sigma"]


def test_requested_parameters_are_returned_sorted(normal: ParametricFamilyDistribution) -> None:
    assert make_block({"sigma", "mu"}).params_to_update(normal) == ["mu", "sigma"]
    assert make_block({"sigma"}).params_to_update(normal) == ["sigma"]


def test_requested_free_parameter_of_view_component(normal_fixed_mu: ParametricFamilyDistribution) -> None:
    assert make_block({"sigma"}).params_to_update(normal_fixed_mu) == ["sigma"]


def test_empty_request_means_nothing_to_update(normal: ParametricFamilyDistribution) -> None:
    assert make_block(set()).params_to_update(normal) == []


def test_unknown_parameter_is_rejected(normal: ParametricFamilyDistribution) -> None:
    with pytest.raises(ValueError, match=r"component 3: \['sigm'\] are not parameters of the 'meanStd'") as error:
        make_block({"mu", "sigm"}).params_to_update(normal)

    assert "Its free parameters are ['mu', 'sigma']" in str(error.value)


def test_parameter_of_another_parametrization_is_rejected(normal: ParametricFamilyDistribution) -> None:
    with pytest.raises(ValueError, match=r"\['var'\] are not parameters of the 'meanStd'"):
        make_block({"var"}).params_to_update(normal)


def test_fixed_parameter_is_rejected(normal_fixed_mu: ParametricFamilyDistribution) -> None:
    with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized") as error:
        make_block({"mu", "sigma"}).params_to_update(normal_fixed_mu)

    assert "Its free parameters are ['sigma']" in str(error.value)


def test_negative_component_id_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"component_id must be non-negative, got -1"):
        OptimizationBlock(-1, None, MaximizationStrategy.QFUNCTION)


def test_non_negative_component_id_is_allowed() -> None:
    assert OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION).component_id == 0

    component_id = 3
    block = OptimizationBlock(component_id, {"mu"}, MaximizationStrategy.MOMENTS)
    assert block.component_id == component_id


def test_unknown_and_fixed_parameters_are_reported_together(normal_fixed_mu: ParametricFamilyDistribution) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"\['tau'\] are not parameters of the 'meanStd' parametrization of 'Normal'; "
            r"\['mu'\] are fixed and cannot be optimized"
        ),
    ):
        make_block({"mu", "tau"}).params_to_update(normal_fixed_mu)
