"""
Internal helpers for working with mixture components, which are `pysatl-core` distributions.

All access to `pysatl-core` distribution internals is kept in this module so that a change of the
`pysatl-core` API affects a single place.

A component created from ``family.view(...)`` stores only its free parameters; the fixed ones live in the
view. `pysatl-core` has no public way to get from such a component to its view
(``component.family`` returns the base family from the registry), so ``_family`` reads it from the
parametrization class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.families.parametric_family import PartialParametricFamily
from pysatl_core.types import CharacteristicName

from ..typings import FloatArray

type ComponentKey = tuple[str, str, tuple[tuple[str, float], ...]]


def lpdf(component: ParametricFamilyDistribution, X: ArrayLike) -> np.float64 | FloatArray:
    """Log-density of a component; points outside support evaluate to -inf."""

    return _evaluate(component, CharacteristicName.LPDF, X)


def pdf(component: ParametricFamilyDistribution, X: ArrayLike) -> np.float64 | FloatArray:
    """Probability density of a component."""

    return _evaluate(component, CharacteristicName.PDF, X)


def free_parameters(component: ParametricFamilyDistribution) -> dict[str, float]:
    """Free (optimizable) parameters of a component in its current parametrization."""

    return dict(component.parameters)


def fixed_parameters(component: ParametricFamilyDistribution) -> dict[str, float]:
    """Parameters fixed via family view, or empty if created without a view."""

    family = _family(component)
    if isinstance(family, PartialParametricFamily):
        return dict(family.fixed_parameters)
    return {}


def all_parameters(component: ParametricFamilyDistribution) -> dict[str, float]:
    """Combined fixed and free parameters of a component."""

    return {**fixed_parameters(component), **free_parameters(component)}


def with_parameters(component: ParametricFamilyDistribution, **updates: float) -> ParametricFamilyDistribution:
    """Rebuild component with updated free parameter values, leaving the original unchanged.

    The new component gets the default sampling and computation strategies, so custom ones of the original
    are not carried over.

    Raises
    ------
    ValueError
        If a name in ``updates`` is not a free parameter of the component, or if the new values violate the
        constraints of the parametrization. The M-step strategies treat the latter as an infeasible point.
    """

    free = free_parameters(component)
    if unknown := set(updates) - set(free):
        raise ValueError(f"Cannot update {sorted(unknown)}: the free parameters of the component are {sorted(free)}.")

    return _family(component).distribution(
        parametrization_name=component.parametrization_name,
        sampling_strategy=None,
        computation_strategy=None,
        **{**free, **updates},
    )


def component_key(component: ParametricFamilyDistribution) -> ComponentKey:
    """Hashable key (family, parametrization, sorted parameter pairs) for equality and hashing."""

    return (
        component.family_name,
        component.parametrization_name,
        tuple(sorted(all_parameters(component).items())),
    )


def _evaluate(
    component: ParametricFamilyDistribution, characteristic: CharacteristicName, X: ArrayLike
) -> np.float64 | FloatArray:
    result = np.asarray(
        component.calculate_characteristic(characteristic, np.asarray(X, dtype=np.float64)), dtype=np.float64
    )
    return result[()] if result.ndim == 0 else result


def _family(component: ParametricFamilyDistribution) -> ParametricFamily:
    """For a component created from ``view`` this is the view itself, otherwise the base family."""

    return type(component.parametrization).__family__
