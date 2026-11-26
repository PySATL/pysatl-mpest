"""Tests for formula dispatch on the family and parametrization of a family view."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pytest
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.initializers._dispatch import FormulaDispatch

from tests.helpers.families import family


@pytest.fixture
def normal() -> ParametricFamily:
    """The Normal family without a view."""

    return family(FamilyName.NORMAL)


@pytest.fixture
def expon() -> ParametricFamily:
    """The Exponential family without a view."""

    return family(FamilyName.EXPONENTIAL)


@pytest.fixture
def formulas() -> FormulaDispatch[str]:
    """A fresh dispatch, so that registrations do not leak between tests."""

    return FormulaDispatch()


def test_dispatches_by_family_and_parametrization(formulas: FormulaDispatch[str], normal: ParametricFamily) -> None:
    formulas.register("Normal", "meanStd")("impl")

    assert formulas.dispatch(normal) == "impl"


@pytest.mark.parametrize("fixed", [{"mu": 0.0}, {"sigma": 1.0}, {}], ids=["fixed-mu", "fixed-sigma", "empty-view"])
def test_view_with_fixed_parameters_dispatches_as_plain_family(
    formulas: FormulaDispatch[str], normal: ParametricFamily, fixed: dict[str, Any]
) -> None:
    formulas.register("Normal", "meanStd")("impl")

    assert formulas.dispatch(normal.view(**fixed)) == "impl"


@pytest.mark.parametrize(
    "view_kwargs",
    [{"parametrization_name": "meanVar"}, {"parametrization_name": "meanVar", "mu": 1.0}],
    ids=["meanVar", "meanVar-fixed-mu"],
)
def test_view_in_other_parametrization_dispatches_by_it(
    formulas: FormulaDispatch[str], normal: ParametricFamily, view_kwargs: dict[str, Any]
) -> None:
    formulas.register("Normal", "meanStd")("meanStd impl")
    formulas.register("Normal", "meanVar")("meanVar impl")

    view = normal.view(**view_kwargs)

    assert formulas.dispatch(view) == "meanVar impl"
    assert formulas.dispatch(normal) == "meanStd impl"


def test_unregistered_parametrization_raises(formulas: FormulaDispatch[str], normal: ParametricFamily) -> None:
    formulas.register("Normal", "meanStd")("impl")

    with pytest.raises(NotImplementedError, match="family 'Normal' and parametrization 'meanVar'"):
        formulas.dispatch(normal.view(parametrization_name="meanVar"))


def test_unregistered_family_raises(formulas: FormulaDispatch[str], expon: ParametricFamily) -> None:
    formulas.register("Normal", "meanStd")("impl")

    with pytest.raises(NotImplementedError, match="family 'Exponential' and parametrization 'rate'"):
        formulas.dispatch(expon)


def test_supports_reports_registration_without_errors(
    formulas: FormulaDispatch[str], normal: ParametricFamily, expon: ParametricFamily
) -> None:
    assert not formulas.supports(normal)

    formulas.register("Normal", "meanStd")("impl")

    assert formulas.supports(normal)
    assert formulas.supports(normal.view(mu=0.0))
    assert not formulas.supports(normal.view(parametrization_name="meanVar"))
    assert not formulas.supports(expon)


def test_family_name_enum_and_string_are_the_same_key(
    formulas: FormulaDispatch[str], normal: ParametricFamily, expon: ParametricFamily
) -> None:
    formulas.register(FamilyName.NORMAL, "meanStd")("normal impl")
    formulas.register("Exponential", "rate")("expon impl")

    assert ("Normal", "meanStd") in formulas.registry
    assert formulas.dispatch(normal) == "normal impl"
    assert formulas.dispatch(expon) == "expon impl"


def test_register_returns_implementation_unchanged(formulas: FormulaDispatch[str]) -> None:
    assert formulas.register("Normal", "meanStd")("impl") == "impl"


def test_register_rejects_duplicate(formulas: FormulaDispatch[str], normal: ParametricFamily) -> None:
    formulas.register("Normal", "meanStd")("first")

    with pytest.raises(ValueError, match="already registered for family 'Normal' and parametrization 'meanStd'"):
        formulas.register(FamilyName.NORMAL, "meanStd")("second")

    assert formulas.dispatch(normal) == "first"


def test_registry_is_read_only_and_reflects_registrations(
    formulas: FormulaDispatch[str],
) -> None:
    formulas.register("Normal", "meanStd")("impl")

    assert dict(formulas.registry) == {("Normal", "meanStd"): "impl"}

    with pytest.raises(TypeError):
        formulas.registry[("X", "y")] = "impl"  # type: ignore[index]


def test_instances_do_not_share_registrations(normal: ParametricFamily, formulas: FormulaDispatch[str]) -> None:
    formulas.register("Normal", "meanStd")("impl")
    other: FormulaDispatch[str] = FormulaDispatch()

    assert not other.supports(normal)
    assert dict(other.registry) == {}
