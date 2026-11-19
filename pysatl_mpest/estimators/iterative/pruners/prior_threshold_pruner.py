"""Pruner for removing mixture components with small weights."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

from ....typings import Scalar
from ..pipeline_state import PipelineState
from ..pruner import Pruner


class PriorThresholdPruner(Pruner):
    """Component pruner based on a prior weight threshold.

    Removes components whose weight is strictly below ``threshold``. Guarantees
    that at least one component remains in the mixture. Remaining weights are
    renormalized.

    Parameters
    ----------
    threshold : Scalar
        Minimum weight to retain a component. Must be strictly between 0 and 1.

    Attributes
    ----------
    threshold : float
        Configured weight threshold.

    Raises
    ------
    ValueError
        If ``threshold`` is not in the exclusive range (0, 1).

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        prune
    """

    def __init__(self, threshold: Scalar) -> None:
        self.threshold = float(threshold)
        if not (0 < self.threshold < 1):
            raise ValueError("Threshold must be between 0 and 1.")

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        """Remove components with weights below threshold, retaining at least one component.

        All components whose weight is strictly below ``threshold`` are removal candidates. If at least
        one weight is greater than or equal to the threshold, every candidate is removed. If every weight
        falls below the threshold, all candidates except the component with the largest weight are
        removed; ties are resolved in favor of the smallest index. The mixture in ``state`` is not
        modified; the returned state holds a new mixture built with ``MixtureModel.without_component``.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state.

        Returns
        -------
        tuple[PipelineState, list[int]]
            Tuple of updated pipeline state and sorted list of removed component indices.
        """

        mixture = state.curr_mixture
        n = mixture.n_components

        to_remove = [i for i, w in enumerate(mixture.weights) if w < self.threshold]

        if len(to_remove) == n:
            to_remove.remove(int(np.argmax(mixture.weights)))

        if not to_remove:
            return (state, [])

        new_mixture = mixture
        for i in sorted(to_remove, reverse=True):
            new_mixture = new_mixture.without_component(i)

        new_state = PipelineState(
            X=state.X, H=state.H, prev_mixture=state.prev_mixture, curr_mixture=new_mixture, error=state.error
        )

        return (new_state, sorted(to_remove))
