"""Expectation-Maximization (EM) estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence

from numpy.typing import ArrayLike

from ..core import MixtureModel
from ..optimizers import Optimizer
from .base_estimator import BaseEstimator
from .iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    Pipeline,
    Pruner,
    WeightsStep,
)
from .iterative._iteration_history import IterationsHistory


class EM(BaseEstimator):
    """Expectation-Maximization (EM) estimator.

    Executes an iterative pipeline of Expectation and Maximization steps.
    Maximization is partitioned component-wise, updating all free parameters
    of each component by maximizing the Q-function.

    Parameters
    ----------
    breakpointers : Sequence[Breakpointer]
        Stopping criteria for the iterative pipeline.
    pruners : Sequence[Pruner]
        Component pruning strategies applied during iterations.
    optimizer : Optimizer
        Numerical optimizer used in maximization steps.

    Attributes
    ----------
    breakpointers : list[Breakpointer]
        Configured stopping criteria.
    pruners : list[Pruner]
        Configured component pruning strategies.
    optimizer : Optimizer
        Numerical optimizer instance.
    history : IterationsHistory
        Recorded iteration snapshots, accessible after calling :meth:`fit`.

    """

    def __init__(
        self,
        breakpointers: Sequence[Breakpointer],
        pruners: Sequence[Pruner],
        optimizer: Optimizer,
    ) -> None:
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer
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

    def fit(self, X: ArrayLike, mixture: MixtureModel, once_in_iterations: int = 1) -> MixtureModel:
        """Fit mixture model parameters using the EM algorithm.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit; not modified in-place.
        once_in_iterations : int, optional
            Iteration recording frequency in history. Defaults to 1.

        Returns
        -------
        MixtureModel
            Fitted mixture model with estimated parameters.
        """

        # No parameter names: every free parameter of a component is optimized, and the parameters
        # fixed with ``family.view(...)`` keep their values.
        blocks = [OptimizationBlock(i, None, MaximizationStrategy.QFUNCTION) for i in range(mixture.n_components)]

        pipeline: Pipeline = Pipeline(
            [ExpectationStep(), WeightsStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )
        result = pipeline.fit(X, mixture)
        self._history = pipeline.history
        return result
