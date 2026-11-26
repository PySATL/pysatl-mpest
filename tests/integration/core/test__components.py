"""
Integration tests for the internal component helpers, on real pysatl-core families.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import (
    ParametricFamily,
    ParametricFamilyDistribution,
    Parametrization,
    parametrization,
)
from pysatl_core.types import CharacteristicName, FamilyName, UnivariateContinuous
from pysatl_mpest.core import _components as cc
from scipy import stats

from tests.helpers.families import family
from tests.helpers.math_assertions import assert_is_array_type, assert_is_scalar_type

X = np.array([-1.0, 0.0, 0.5, 2.0])


@pytest.fixture
def normal() -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=0.5, sigma=2.0)


@pytest.fixture
def normal_fixed_mu() -> ParametricFamilyDistribution:
    """``mu`` is fixed to 1.0 through a view, ``sigma`` is free."""
    return family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)


@pytest.fixture
def normal_mean_var_fixed_mu() -> ParametricFamilyDistribution:
    """A view in the ``meanVar`` parametrization: ``mu`` is fixed, ``var`` is free."""
    return family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0)(var=4.0)


class TestEvaluation:
    def test_lpdf_matches_scipy(self, normal: ParametricFamilyDistribution) -> None:
        assert_allclose(cc.lpdf(normal, X), stats.norm(loc=0.5, scale=2.0).logpdf(X), rtol=1e-12)

    def test_pdf_matches_scipy(self, normal: ParametricFamilyDistribution) -> None:
        assert_allclose(cc.pdf(normal, X), stats.norm(loc=0.5, scale=2.0).pdf(X), rtol=1e-12)

    def test_preserves_scalar_and_array_shapes(self, normal: ParametricFamilyDistribution) -> None:
        assert_is_scalar_type(cc.lpdf(normal, 0.5))
        assert_is_scalar_type(cc.pdf(normal, 0.5))
        assert_is_array_type(cc.lpdf(normal, X), X.shape)
        assert_is_array_type(cc.pdf(normal, np.zeros((2, 3))), (2, 3))

    def test_lpdf_is_neg_inf_outside_support(self) -> None:
        exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.5)

        assert_allclose(cc.lpdf(exponential, np.array([-1.0, 0.0])), [-np.inf, np.log(1.5)])
        assert_allclose(cc.pdf(exponential, np.array([-1.0])), [0.0])


class TestParameters:
    def test_returns_plain_component_parameters(self, normal: ParametricFamilyDistribution) -> None:
        assert cc.free_parameters(normal) == {"mu": 0.5, "sigma": 2.0}
        assert cc.fixed_parameters(normal) == {}
        assert cc.all_parameters(normal) == {"mu": 0.5, "sigma": 2.0}

    def test_returns_view_component_parameters(self, normal_fixed_mu: ParametricFamilyDistribution) -> None:
        assert cc.free_parameters(normal_fixed_mu) == {"sigma": 2.0}
        assert cc.fixed_parameters(normal_fixed_mu) == {"mu": 1.0}
        assert cc.all_parameters(normal_fixed_mu) == {"mu": 1.0, "sigma": 2.0}

    def test_returns_view_parameters_in_non_base_parametrization(
        self, normal_mean_var_fixed_mu: ParametricFamilyDistribution
    ) -> None:
        assert cc.free_parameters(normal_mean_var_fixed_mu) == {"var": 4.0}
        assert cc.fixed_parameters(normal_mean_var_fixed_mu) == {"mu": 1.0}
        assert cc.all_parameters(normal_mean_var_fixed_mu) == {"mu": 1.0, "var": 4.0}


class TestWithParameters:
    def test_updates_plain_component(self, normal: ParametricFamilyDistribution) -> None:
        updated = cc.with_parameters(normal, mu=-1.0)

        assert cc.all_parameters(updated) == {"mu": -1.0, "sigma": 2.0}
        assert updated.parametrization_name == normal.parametrization_name

    def test_does_not_change_original(self, normal: ParametricFamilyDistribution) -> None:
        cc.with_parameters(normal, sigma=5.0)

        assert cc.all_parameters(normal) == {"mu": 0.5, "sigma": 2.0}

    def test_keeps_fixed_parameters(self, normal_fixed_mu: ParametricFamilyDistribution) -> None:
        updated = cc.with_parameters(normal_fixed_mu, sigma=3.0)

        assert cc.fixed_parameters(updated) == {"mu": 1.0}
        assert cc.free_parameters(updated) == {"sigma": 3.0}
        assert_allclose(cc.lpdf(updated, X), stats.norm(loc=1.0, scale=3.0).logpdf(X), rtol=1e-12)

    def test_keeps_parametrization(self, normal_mean_var_fixed_mu: ParametricFamilyDistribution) -> None:
        updated = cc.with_parameters(normal_mean_var_fixed_mu, var=9.0)

        assert updated.parametrization_name == "meanVar"
        assert cc.all_parameters(updated) == {"mu": 1.0, "var": 9.0}
        assert_allclose(cc.lpdf(updated, X), stats.norm(loc=1.0, scale=3.0).logpdf(X), rtol=1e-12)

    def test_without_updates_is_equivalent(self, normal_fixed_mu: ParametricFamilyDistribution) -> None:
        same = cc.with_parameters(normal_fixed_mu)

        assert same is not normal_fixed_mu
        assert cc.component_key(same) == cc.component_key(normal_fixed_mu)

    def test_rejects_fixed_parameter(self, normal_fixed_mu: ParametricFamilyDistribution) -> None:
        with pytest.raises(ValueError, match=r"Cannot update \['mu'\]"):
            cc.with_parameters(normal_fixed_mu, mu=0.0)

    def test_rejects_unknown_parameter(self, normal: ParametricFamilyDistribution) -> None:
        with pytest.raises(ValueError, match=r"Cannot update \['tau'\]"):
            cc.with_parameters(normal, tau=1.0)

    def test_rejects_constraint_violation(self, normal_fixed_mu: ParametricFamilyDistribution) -> None:
        with pytest.raises(ValueError, match="sigma > 0"):
            cc.with_parameters(normal_fixed_mu, sigma=-1.0)


class TestComponentKey:
    def test_equal_for_equal_components(self) -> None:
        first = family(FamilyName.NORMAL)(mu=0.0, sigma=1.0)
        second = family(FamilyName.NORMAL)(mu=0.0, sigma=1.0)

        assert cc.component_key(first) == cc.component_key(second)
        assert hash(cc.component_key(first)) == hash(cc.component_key(second))

    def test_equal_for_independent_views(self) -> None:
        first = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)
        second = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)

        assert cc.component_key(first) == cc.component_key(second)

    def test_view_equals_full_component(self, normal_fixed_mu: ParametricFamilyDistribution) -> None:
        full = family(FamilyName.NORMAL)(mu=1.0, sigma=2.0)

        assert cc.component_key(normal_fixed_mu) == cc.component_key(full)

    def test_differs_by_parameter_value(self, normal: ParametricFamilyDistribution) -> None:
        assert cc.component_key(normal) != cc.component_key(cc.with_parameters(normal, sigma=2.5))

    def test_differs_by_fixed_value(self) -> None:
        first = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)
        second = family(FamilyName.NORMAL).view(mu=2.0)(sigma=2.0)

        assert cc.component_key(first) != cc.component_key(second)

    def test_differs_by_family(self) -> None:
        uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
        exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.0)

        assert cc.component_key(uniform) != cc.component_key(exponential)

    def test_differs_by_parametrization(self) -> None:
        mean_std = family(FamilyName.NORMAL)(parametrization_name="meanStd", mu=0.0, sigma=2.0)
        mean_var = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=4.0)

        assert cc.component_key(mean_std) != cc.component_key(mean_var)

    def test_is_usable_as_dict_key(self, normal: ParametricFamilyDistribution) -> None:
        registry = {cc.component_key(normal): "normal"}

        assert registry[cc.component_key(cc.with_parameters(normal))] == "normal"


def three_parameter_family() -> ParametricFamily:
    """Build an unregistered family with three parameters: a view of a built-in family fixes at most one."""

    result = ParametricFamily(
        name="ThreeParameter",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["abc"],
        distr_characteristics={CharacteristicName.PDF: lambda parameters, x: np.ones_like(x)},
    )

    @parametrization(family=result, name="abc")
    class _Abc(Parametrization):
        a: float
        b: float
        c: float

    return result


FAMILY_CASES: dict[str, tuple[Callable[[], ParametricFamily], dict[str, float]]] = {
    "normal": (lambda: family(FamilyName.NORMAL), {"mu": 0.5, "sigma": 2.0}),
    "exponential": (lambda: family(FamilyName.EXPONENTIAL), {"lambda_": 1.5}),
    "uniform": (lambda: family(FamilyName.CONTINUOUS_UNIFORM), {"lower_bound": 0.0, "upper_bound": 1.0}),
    "normal-fixed-mu": (lambda: family(FamilyName.NORMAL).view(mu=1.0), {"sigma": 2.0}),
    "normal-meanVar": (
        lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar"),
        {"mu": 0.0, "var": 4.0},
    ),
    "normal-meanVar-fixed-mu": (
        lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0),
        {"var": 4.0},
    ),
    "uniform-fixed-lower": (lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(lower_bound=0.0), {"upper_bound": 1.0}),
}

family_cases = pytest.mark.parametrize("make_family, free", FAMILY_CASES.values(), ids=FAMILY_CASES.keys())


class TestFamilyKey:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param(FamilyName.NORMAL, ("Normal", "meanStd", ()), id="normal"),
            pytest.param(FamilyName.EXPONENTIAL, ("Exponential", "rate", ()), id="exponential"),
            pytest.param(FamilyName.CONTINUOUS_UNIFORM, ("ContinuousUniform", "standard", ()), id="uniform"),
        ],
    )
    def test_plain_family_has_no_fixed_parameters(self, name: FamilyName, expected: cc.FamilyKey) -> None:
        assert cc.family_key(family(name)) == expected

    def test_view_contains_fixed_parameter(self) -> None:
        assert cc.family_key(family(FamilyName.NORMAL).view(mu=0.0)) == ("Normal", "meanStd", (("mu", 0.0),))

    def test_view_contains_its_parametrization(self) -> None:
        assert cc.family_key(family(FamilyName.NORMAL).view(parametrization_name="meanVar")) == (
            "Normal",
            "meanVar",
            (),
        )
        assert cc.family_key(family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0)) == (
            "Normal",
            "meanVar",
            (("mu", 1.0),),
        )

    def test_view_without_fixed_parameters_in_base_parametrization_has_plain_family_key(self) -> None:
        normal_family = family(FamilyName.NORMAL)

        assert cc.family_key(normal_family.view()) == cc.family_key(normal_family)
        assert cc.family_key(normal_family.view(parametrization_name="meanStd")) == cc.family_key(normal_family)

    def test_differs_by_fixed_value(self) -> None:
        assert cc.family_key(family(FamilyName.NORMAL).view(mu=0.0)) != cc.family_key(
            family(FamilyName.NORMAL).view(mu=5.0)
        )

    def test_differs_by_fixed_parameter(self) -> None:
        assert cc.family_key(family(FamilyName.NORMAL).view(mu=1.0)) != cc.family_key(
            family(FamilyName.NORMAL).view(sigma=1.0)
        )

    def test_differs_by_parametrization(self) -> None:
        assert cc.family_key(family(FamilyName.NORMAL)) != cc.family_key(
            family(FamilyName.NORMAL).view(parametrization_name="meanVar")
        )

    def test_differs_by_family(self) -> None:
        assert cc.family_key(family(FamilyName.CONTINUOUS_UNIFORM)) != cc.family_key(family(FamilyName.EXPONENTIAL))

    def test_equal_for_independent_equal_views(self) -> None:
        first = family(FamilyName.NORMAL).view(mu=1.0)
        second = family(FamilyName.NORMAL).view(mu=1.0)

        assert cc.family_key(first) == cc.family_key(second)
        assert hash(cc.family_key(first)) == hash(cc.family_key(second))

    def test_int_and_float_fixed_values_give_equal_keys(self) -> None:
        first = family(FamilyName.NORMAL).view(mu=0)
        second = family(FamilyName.NORMAL).view(mu=0.0)

        assert cc.family_key(first) == cc.family_key(second)
        assert hash(cc.family_key(first)) == hash(cc.family_key(second))

    def test_fixed_parameters_sorted_by_name(self) -> None:
        three = three_parameter_family()

        assert (
            cc.family_key(three.view(a=1.0, b=2.0))
            == cc.family_key(three.view(b=2.0, a=1.0))
            == (
                "ThreeParameter",
                "abc",
                (("a", 1.0), ("b", 2.0)),
            )
        )


FIXED_PARAMETER_CASES: dict[str, tuple[Callable[[], ParametricFamily], dict[str, float]]] = {
    "plain": (lambda: family(FamilyName.NORMAL), {}),
    "fixed-mu": (lambda: family(FamilyName.NORMAL).view(mu=1.0), {"mu": 1.0}),
    "meanVar": (lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar"), {}),
    "meanVar-fixed-var": (
        lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar", var=4.0),
        {"var": 4.0},
    ),
}

fixed_parameter_cases = pytest.mark.parametrize(
    "make_family, expected", FIXED_PARAMETER_CASES.values(), ids=FIXED_PARAMETER_CASES.keys()
)

FREE_PARAMETER_CASES: dict[str, tuple[Callable[[], ParametricFamily], tuple[str, ...]]] = {
    "normal": (lambda: family(FamilyName.NORMAL), ("mu", "sigma")),
    "exponential": (lambda: family(FamilyName.EXPONENTIAL), ("lambda_",)),
    "uniform": (lambda: family(FamilyName.CONTINUOUS_UNIFORM), ("lower_bound", "upper_bound")),
    "fixed-mu": (lambda: family(FamilyName.NORMAL).view(mu=0.0), ("sigma",)),
    "fixed-sigma": (lambda: family(FamilyName.NORMAL).view(sigma=1.0), ("mu",)),
    "meanVar": (lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar"), ("mu", "var")),
    "meanVar-fixed-var": (
        lambda: family(FamilyName.NORMAL).view(parametrization_name="meanVar", var=4.0),
        ("mu",),
    ),
    "uniform-fixed-upper": (lambda: family(FamilyName.CONTINUOUS_UNIFORM).view(upper_bound=3.0), ("lower_bound",)),
}

free_parameter_cases = pytest.mark.parametrize(
    "make_family, expected", FREE_PARAMETER_CASES.values(), ids=FREE_PARAMETER_CASES.keys()
)


class TestFamilyParameters:
    @fixed_parameter_cases
    def test_returns_fixed_parameters(
        self, make_family: Callable[[], ParametricFamily], expected: dict[str, float]
    ) -> None:
        assert cc.family_fixed_parameters(make_family()) == expected

    def test_returns_new_dictionary_each_call(self) -> None:
        view = family(FamilyName.NORMAL).view(mu=1.0)

        first = cc.family_fixed_parameters(view)
        first["mu"] = 5.0

        assert cc.family_fixed_parameters(view) == {"mu": 1.0}

    @free_parameter_cases
    def test_returns_free_parameter_names_in_declaration_order(
        self, make_family: Callable[[], ParametricFamily], expected: tuple[str, ...]
    ) -> None:
        assert cc.family_free_parameter_names(make_family()) == expected

    def test_free_and_fixed_partition_parametrization_parameters(self) -> None:
        view = family(FamilyName.NORMAL).view(mu=1.0)

        free = set(cc.family_free_parameter_names(view))
        fixed = set(cc.family_fixed_parameters(view))

        assert free | fixed == {"mu", "sigma"}
        assert not free & fixed

    @family_cases
    def test_family_parameters_match_component(
        self, make_family: Callable[[], ParametricFamily], free: dict[str, float]
    ) -> None:
        family_ = make_family()
        component = family_.distribution(**free)

        assert cc.family_free_parameter_names(family_) == tuple(cc.free_parameters(component))
        assert cc.family_fixed_parameters(family_) == cc.fixed_parameters(component)


class TestComponentFamilyKey:
    @family_cases
    def test_equals_family_key(self, make_family: Callable[[], ParametricFamily], free: dict[str, float]) -> None:
        family_ = make_family()

        assert cc.component_family_key(family_.distribution(**free)) == cc.family_key(family_)

    def test_ignores_free_parameter_values(self) -> None:
        assert cc.component_family_key(family(FamilyName.NORMAL)(mu=0.0, sigma=1.0)) == cc.component_family_key(
            family(FamilyName.NORMAL)(mu=5.0, sigma=2.0)
        )

    def test_view_component_differs_from_full_component(self) -> None:
        view_component = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)
        full_component = family(FamilyName.NORMAL)(mu=1.0, sigma=2.0)

        assert cc.component_family_key(view_component) != cc.component_family_key(full_component)

    def test_survives_with_parameters(self) -> None:
        component = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)

        assert cc.component_family_key(cc.with_parameters(component, sigma=3.0)) == cc.component_family_key(component)

    def test_component_in_non_base_parametrization_gets_view_key(self) -> None:
        component = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=4.0)

        assert (
            cc.component_family_key(component)
            == cc.family_key(family(FamilyName.NORMAL).view(parametrization_name="meanVar"))
            == ("Normal", "meanVar", ())
        )
        assert cc.component_family_key(component) != cc.family_key(family(FamilyName.NORMAL))
