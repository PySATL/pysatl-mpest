"""Tests for family and parametrization dispatch behavior."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.estimators.iterative._strategies._dispatch import FamilyDispatch

from tests.mocks.distributions import MockContinuousDistribution


def make_component(
    family_name: str = "MockFamily", parametrization_name: str = "MockParam"
) -> ParametricFamilyDistribution:
    mock = MockContinuousDistribution()
    mock.family_name = family_name
    mock.parametrization_name = parametrization_name
    return cast(ParametricFamilyDistribution, mock)


def default_implementation(component: ParametricFamilyDistribution, value: int) -> str:
    """Default implementation."""
    return f"default {value}"


@pytest.fixture
def strategy() -> FamilyDispatch[[int], str]:
    """A fresh dispatch with a default implementation, so that registrations do not leak between tests."""

    return FamilyDispatch(default_implementation, name="strategy", doc="Strategy docstring.")


def test_uses_default_when_nothing_is_registered(strategy: FamilyDispatch[[int], str]) -> None:
    assert strategy(make_component(), 1) == "default 1"


def test_uses_registered_implementation_for_family_and_parametrization(strategy: FamilyDispatch[[int], str]) -> None:
    @strategy.register("MockFamily", "MockParam")
    def registered(component: ParametricFamilyDistribution, value: int) -> str:
        return f"registered {value}"

    assert strategy(make_component(), 2) == "registered 2"


def test_passes_arguments_to_implementation(strategy: FamilyDispatch[[int], str]) -> None:
    @strategy.register("MockFamily", "MockParam")
    def registered(component: ParametricFamilyDistribution, value: int) -> str:
        return f"{component.family_name} {value}"

    assert strategy(make_component(), value=3) == "MockFamily 3"


def test_other_parametrization_of_registered_family_uses_default(strategy: FamilyDispatch[[int], str]) -> None:
    strategy.register("MockFamily", "MockParam")(lambda component, value: "registered")

    assert strategy(make_component(parametrization_name="Other"), 4) == "default 4"


def test_other_family_uses_default(strategy: FamilyDispatch[[int], str]) -> None:
    strategy.register("MockFamily", "MockParam")(lambda component, value: "registered")

    assert strategy(make_component(family_name="Other"), 5) == "default 5"


def test_family_name_enum_and_string_are_the_same_key(strategy: FamilyDispatch[[int], str]) -> None:
    strategy.register(FamilyName.NORMAL, "meanStd")(lambda component, value: "normal")

    assert strategy(make_component(family_name="Normal", parametrization_name="meanStd"), 6) == "normal"
    assert ("Normal", "meanStd") in strategy.registry


def test_register_returns_function_unchanged(strategy: FamilyDispatch[[int], str]) -> None:
    def registered(component: ParametricFamilyDistribution, value: int) -> str:
        return "registered"

    assert strategy.register("MockFamily", "MockParam")(registered) is registered


def test_register_rejects_duplicate(strategy: FamilyDispatch[[int], str]) -> None:
    strategy.register("MockFamily", "MockParam")(lambda component, value: "first")

    with pytest.raises(ValueError, match="already registered for family 'MockFamily' and parametrization 'MockParam'"):
        strategy.register("MockFamily", "MockParam")(lambda component, value: "second")


def test_dispatch_returns_the_selected_implementation(strategy: FamilyDispatch[[int], str]) -> None:
    def registered(component: ParametricFamilyDistribution, value: int) -> str:
        return "registered"

    strategy.register("MockFamily", "MockParam")(registered)

    assert strategy.dispatch(make_component()) is registered
    assert strategy.dispatch(make_component(family_name="Other")) is default_implementation


def test_registry_is_read_only(strategy: FamilyDispatch[[int], str]) -> None:
    with pytest.raises(TypeError):
        strategy.registry[("MockFamily", "MockParam")] = lambda component, value: "registered"  # type: ignore[index]


def test_has_explicit_name_qualname_and_doc(strategy: FamilyDispatch[[int], str]) -> None:
    assert strategy.__name__ == "strategy"
    assert strategy.__qualname__ == "strategy"
    assert strategy.__doc__ == "Strategy docstring."
    assert not hasattr(strategy, "__wrapped__")
