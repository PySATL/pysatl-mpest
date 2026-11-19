"""
Integration tests for the internal component helpers, on real pysatl-core families.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
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
