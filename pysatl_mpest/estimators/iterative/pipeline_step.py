"""Abstract base class for iterative pipeline processing steps."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from .pipeline_state import PipelineState


class PipelineStep(ABC):
    """Abstract base class for a single processing step in an iterative pipeline.

    Attributes
    ----------
    available_next_steps : list[type[PipelineStep]]
        Valid step types that may immediately follow this step.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        run
        clear_after_prune
        reset
    """

    @property
    @abstractmethod
    def available_next_steps(self) -> list[type["PipelineStep"]]:
        """list[type[PipelineStep]]: Step types permitted to immediately follow this step."""

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute the step on the given pipeline state and return the updated state.

        The built-in steps modify ``state`` in place and return it. A step may report a failure by setting
        ``state.error``, which stops the pipeline with a warning.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state.

        Returns
        -------
        PipelineState
            Updated pipeline state.
        """

    @abstractmethod
    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """Clean up step internal state for components removed during pruning.

        Surviving components are renumbered contiguously starting from 0; step
        state indexed by component must be re-indexed accordingly. The
        re-indexed state stays in effect until :meth:`reset` is called.

        Parameters
        ----------
        removed_components_indices : list[int]
            Ascending indices of removed components relative to the mixture before this removal.
        """

    @abstractmethod
    def reset(self) -> None:
        """Return the step to its post-constructor state.

        :meth:`Pipeline.fit` calls this method on every step before the first
        iteration, so no per-fit state carries over between fits. Steps without
        such state may leave the body empty.
        """
