"""Maximization step strategies and optimization block configurations."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from dataclasses import dataclass
from enum import Enum, auto

from pysatl_core.families import ParametricFamilyDistribution

from ....core import _components


class MaximizationStrategy(Enum):
    """Available parameter update strategies for :class:`MaximizationStep`.

    Attributes
    ----------
    QFUNCTION
        Maximize conditional expectation of complete-data log-likelihood (Q-function).
    OBSERVED_DATA_LIKELIHOOD
        Directly maximize observed-data log-likelihood of the mixture.
    MOMENTS
        Method of Moments matching theoretical and empirical weighted moments. Implemented only for
        the Normal (``meanStd``) and Exponential (``rate``) families; other components raise
        ``NotImplementedError``.
    """

    QFUNCTION = auto()
    OBSERVED_DATA_LIKELIHOOD = auto()
    MOMENTS = auto()


@dataclass
class OptimizationBlock:
    """Configuration task for optimizing a mixture component in :class:`MaximizationStep`.

    Parameters
    ----------
    component_id : int
        Zero-based index of the component to update.
    params_to_optimize : set[str] | None
        Parameter names to optimize, or None to optimize all free parameters.
        Parameters fixed via ``family.view(...)`` cannot be listed here.
    maximization_strategy : MaximizationStrategy
        Objective strategy to apply.
    """

    component_id: int
    params_to_optimize: set[str] | None
    maximization_strategy: MaximizationStrategy

    def __post_init__(self) -> None:
        if self.component_id < 0:
            raise ValueError(f"component_id must be non-negative, got {self.component_id}")

    def params_to_update(self, component: ParametricFamilyDistribution) -> list[str]:
        """Return sorted names of free parameters to optimize for the component.

        Parameters
        ----------
        component : ParametricFamilyDistribution
            Target component.

        Returns
        -------
        list[str]
            Sorted parameter names to update.

        Raises
        ------
        ValueError
            If any parameter is unknown or fixed by a view.
        """

        free = set(_components.free_parameters(component))
        if self.params_to_optimize is None:
            return sorted(free)

        not_free = self.params_to_optimize - free
        if not_free:
            fixed = not_free & set(_components.fixed_parameters(component))
            unknown = not_free - fixed
            problems = []
            if unknown:
                problems.append(
                    f"{sorted(unknown)} are not parameters of the '{component.parametrization_name}' "
                    f"parametrization of '{component.family_name}'"
                )
            if fixed:
                problems.append(f"{sorted(fixed)} are fixed and cannot be optimized")
            raise ValueError(
                f"Cannot optimize parameters of component {self.component_id}: {'; '.join(problems)}. "
                f"Its free parameters are {sorted(free)}."
            )

        return sorted(self.params_to_optimize)
