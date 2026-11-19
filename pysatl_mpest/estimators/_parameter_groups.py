"""Shared validation of parameter groups and construction of optimization blocks.

``parameter_groups`` is the facade-level way to request conditional maximization: for a component it
is an ordered list of parameter groups, and each group becomes a separate maximization block. This
module is shared by the ECM and ECME facades so that the validation rules do not diverge.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence

from pysatl_core.families import ParametricFamilyDistribution

from ..core import MixtureModel, _components
from .iterative import MaximizationStrategy, OptimizationBlock


def build_optimization_blocks(
    mixture: MixtureModel,
    parameter_groups: Mapping[int, Sequence[set[str]]] | None,
    strategies: Sequence[MaximizationStrategy],
) -> list[OptimizationBlock]:
    """Validate parameter groups and build ordered component blocks.

    Parameters
    ----------
    mixture : MixtureModel
        Mixture whose components are optimized.
    parameter_groups : Mapping[int, Sequence[set[str]]] | None
        Ordered parameter groups by component, or None for joint steps.
    strategies : Sequence[MaximizationStrategy]
        Strategy for each component, in component order.

    Returns
    -------
    list[OptimizationBlock]
        Blocks in component and group order, with copied group sets.

    Raises
    ------
    ValueError
        If the number of strategies differs from the number of components,
        or if parameter groups are invalid.
    """

    if len(strategies) != mixture.n_components:
        raise ValueError("The number of strategies must match the number of components.")

    _validate_parameter_groups(mixture, parameter_groups)

    blocks: list[OptimizationBlock] = []
    for component_id, strategy in enumerate(strategies):
        groups = None if parameter_groups is None else parameter_groups.get(component_id)
        if groups is None:
            blocks.append(OptimizationBlock(component_id, None, strategy))
        else:
            blocks.extend(OptimizationBlock(component_id, set(group), strategy) for group in groups)
    return blocks


def _validate_parameter_groups(
    mixture: MixtureModel,
    parameter_groups: Mapping[int, Sequence[set[str]]] | None,
) -> None:
    """Validate component indices and groups against each component's free parameters."""

    if parameter_groups is None:
        return

    n_components = mixture.n_components
    for component_id, groups in parameter_groups.items():
        if component_id < 0:
            raise ValueError(
                f"parameter_groups key {component_id} is negative; component indices must be non-negative."
            )
        if component_id >= n_components:
            raise ValueError(
                f"parameter_groups key {component_id} is out of bounds for a mixture with {n_components} components."
            )
        _validate_component_groups(mixture.components[component_id], component_id, groups)


def _validate_component_groups(
    component: ParametricFamilyDistribution, component_id: int, groups: Sequence[set[str]]
) -> None:
    if len(groups) == 0:
        raise ValueError(f"parameter_groups[{component_id}]: the component's group list must not be empty.")

    for group in groups:
        if len(group) == 0:
            raise ValueError(
                f"parameter_groups[{component_id}] contains an empty group; every group must be non-empty."
            )

    union: set[str] = set()
    for group in groups:
        shared = union & group
        if shared:
            raise ValueError(f"parameter_groups[{component_id}] groups overlap on parameters {sorted(shared)}.")
        union |= group

    free = set(_components.free_parameters(component))
    if union == free:
        return

    extra = union - free
    fixed = extra & set(_components.fixed_parameters(component))
    unknown = extra - fixed
    uncovered = free - union

    problems = []
    if unknown:
        problems.append(
            f"{sorted(unknown)} are not parameters of the '{component.parametrization_name}' "
            f"parametrization of '{component.family_name}'"
        )
    if fixed:
        problems.append(f"{sorted(fixed)} are fixed and cannot be optimized")
    if uncovered:
        problems.append(f"{sorted(uncovered)} are not covered by any group")

    raise ValueError(
        f"Invalid parameter_groups for component {component_id}: {'; '.join(problems)}. "
        f"Its free parameters are {sorted(free)}."
    )
