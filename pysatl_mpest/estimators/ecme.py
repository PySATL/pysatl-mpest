"""Expectation-Conditional Maximization Either (ECME) estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Mapping, Sequence
from typing import Literal

from numpy.typing import ArrayLike

from ..core import MixtureModel
from ..optimizers import Optimizer
from ._parameter_groups import build_optimization_blocks
from .base_estimator import BaseEstimator
from .iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    Pipeline,
    Pruner,
    WeightsStep,
)
from .iterative._iteration_history import IterationsHistory


class ECME(BaseEstimator):
    """Expectation-Conditional Maximization Either (ECME) estimator.

    Optimizes free parameters using the Q-function or observed-data likelihood
    (ODL), preserving fixed parameters. Executes Q-function blocks before ODL
    blocks, each in ascending component order.

    With soft expectations, no pruning, and an optimizer that never worsens its
    objective, observed-data log-likelihood is non-decreasing across iterations.
    :class:`~pysatl_mpest.optimizers.ScipyNelderMead` meets this optimizer condition;
    the base :class:`~pysatl_mpest.optimizers.Optimizer` contract does not require it.

    Parameters
    ----------
    breakpointers : Sequence[Breakpointer]
        Stopping criteria for the iterative pipeline.
    pruners : Sequence[Pruner]
        Component pruning strategies applied during iterations.
    optimizer : Optimizer
        Numerical optimizer used in maximization steps.
    default_strategy : Literal["q-func", "odl"], optional
        Default strategy for components not specified in :meth:`fit`. Defaults to "odl".

    Attributes
    ----------
    breakpointers : list[Breakpointer]
        Configured stopping criteria.
    pruners : list[Pruner]
        Configured component pruning strategies.
    optimizer : Optimizer
        Numerical optimizer instance.
    default_strategy_name : str
        Name of the default strategy ("q-func" or "odl").
    history : IterationsHistory
        Recorded iteration snapshots, accessible after calling :meth:`fit`.

    Raises
    ------
    ValueError
        If ``default_strategy`` is not ``'odl'`` or ``'q-func'``.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    def __init__(
        self,
        breakpointers: Sequence[Breakpointer],
        pruners: Sequence[Pruner],
        optimizer: Optimizer,
        default_strategy: Literal["q-func", "odl"] = "odl",
    ):
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer

        if default_strategy not in ("odl", "q-func"):
            raise ValueError(f"Unknown default_strategy '{default_strategy}'. Expected 'odl' or 'q-func'.")

        self.default_strategy_name = default_strategy

        self._history: IterationsHistory | None = None

    @property
    def history(self) -> IterationsHistory:
        """IterationsHistory: Recorded history of pipeline iterations after fitting.

        Raises
        ------
        AttributeError
            If accessed before ``fit`` has been called.
        """

        if self._history is None:
            raise AttributeError("History is not available. Call the 'fit' method first.")
        return self._history

    def _normalize_indices(self, indices: Sequence[int] | int | None) -> set[int]:
        if indices is None:
            return set()
        if isinstance(indices, int):
            return {indices}
        return set(indices)

    def _validate_indices(self, indices: set[int], n_components: int) -> None:
        for idx in indices:
            if idx < 0:
                raise ValueError(f"Component index {idx} is negative; component indices must be non-negative.")
            if idx >= n_components:
                raise ValueError(f"Component index {idx} is out of bounds for mixture size {n_components}.")

    def fit(
        self,
        X: ArrayLike,
        mixture: MixtureModel,
        q_indices_raw: Sequence[int] | int | None = None,
        odl_indices_raw: Sequence[int] | int | None = None,
        once_in_iterations: int = 1,
        *,
        parameter_groups: Mapping[int, Sequence[set[str]]] | None = None,
    ) -> MixtureModel:
        """Fit mixture model parameters using the ECME algorithm.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit; not modified in-place.
        q_indices_raw : Sequence[int] | int | None, optional
            Component indices to optimize via Q-function maximization. Must be non-negative.
        odl_indices_raw : Sequence[int] | int | None, optional
            Component indices to optimize via ODL maximization. Must be non-negative.
        once_in_iterations : int, optional
            Iteration recording frequency in history. Defaults to 1.
        parameter_groups : Mapping[int, Sequence[set[str]]] | None, optional
            Component indices mapped to ordered, non-empty, disjoint groups covering
            all free parameters. Groups are optimized sequentially with the component's
            strategy; omitted components use one joint step. Defaults to None.

        Returns
        -------
        MixtureModel
            Fitted mixture model with estimated parameters.

        Raises
        ------
        ValueError
            If a component index is invalid or assigned both strategies, or its
            groups do not partition its free parameters into non-empty, disjoint sets.
        """

        n_components = mixture.n_components

        q_indices = self._normalize_indices(q_indices_raw)
        odl_indices = self._normalize_indices(odl_indices_raw)
        self._validate_indices(q_indices, n_components)
        self._validate_indices(odl_indices, n_components)

        intersection = q_indices.intersection(odl_indices)
        if intersection:
            raise ValueError(f"Indices {intersection} specified for both Q-function and ODL strategies.")

        default_enum = (
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            if self.default_strategy_name == "odl"
            else MaximizationStrategy.QFUNCTION
        )

        strategies = [default_enum] * n_components
        for index in q_indices:
            strategies[index] = MaximizationStrategy.QFUNCTION
        for index in odl_indices:
            strategies[index] = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD

        blocks = build_optimization_blocks(mixture, parameter_groups, strategies)
        blocks.sort(key=lambda block: block.maximization_strategy is not MaximizationStrategy.QFUNCTION)

        pipeline: Pipeline = Pipeline(
            [ExpectationStep(), WeightsStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )

        result = pipeline.fit(X, mixture)
        self._history = pipeline.history
        return result
