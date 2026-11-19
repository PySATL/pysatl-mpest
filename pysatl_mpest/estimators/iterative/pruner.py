"""Abstract base class for mixture pruning strategies."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from .pipeline_state import PipelineState


class Pruner(ABC):
    """Abstract base class for component pruning strategies.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        prune
    """

    @abstractmethod
    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        """Identify and remove components from the mixture in pipeline state.

        Implementations return the state to continue with and may build a new one instead of
        modifying the given state. Removed indices refer to the mixture given to this call; the pipeline
        drops the matching columns of ``state.H`` itself.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state.

        Returns
        -------
        tuple[PipelineState, list[int]]
            Tuple of updated pipeline state and sorted indices of removed components.
        """
