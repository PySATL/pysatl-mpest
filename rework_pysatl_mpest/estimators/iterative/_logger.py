"""Provides a container for recording iteration history in the iterative estimation pipeline.

This module defines the `IterationsHistory` class for
storing and accessing detailed snapshots of pipeline execution,
including mixture models, input data, responsibilities, pruning actions,
and errors across iterations.
"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Optional

from numpy import float64
from numpy.typing import NDArray

from ...core import MixtureModel
from .pruner import Pruner


@dataclass
class IterationRecord:
    """Data class representing a single pipeline iteration snapshot.

    This class captures the complete state of a pipeline iteration after
    pruning has been applied, providing a snapshot for analysis, debugging,
    or post-hoc inspection.

    Attributes
    ----------
    iteration : int
        The iteration number (0-based index).
    mixture : MixtureModel
        The state of the mixture model after pruning in this iteration.
    X : NDArray[float64]
        The input data sample being processed (conventionally named `X`).
    H : Optional[NDArray[float64]]
        The responsibility matrix (posterior probabilities) if available,
        where `H[i, j]` represents the probability that data point `i`
        belongs to component `j`. May be `None` if not computed.
    pruners_used : Optional[list[Pruner]]
        List of pruner instances that were applied during this iteration.
        `None` or empty if no pruning occurred.
    error : Optional[Exception]
        Any exception that occurred during the iteration, or `None` if
        the iteration completed successfully.
    """

    iteration: int
    mixture: MixtureModel
    X: NDArray[float64]
    H: Optional[NDArray[float64]]
    pruners_used: Optional[list[Pruner]]
    error: Optional[Exception]


class IterationsHistory:
    """A container for storing and accessing pipeline iteration history.

    `IterationsHistory` collects and stores snapshots of each pipeline iteration
    (as `IterationRecord` objects) according to a configurable frequency.
    It is a structured history buffer for programmatic analysis.

    This class supports sequence-like access via indexing (e.g., `history[0]`)
    and length queries (`len(history)`), making it easy to inspect specific
    iterations or iterate over recorded states.

    Example usage:

    >>> history = IterationsHistory(once_in_iterations=2)
    >>> # Inside pipeline loop:
    >>> record = IterationRecord(iteration=0, mixture=..., X=X, H=H, ...)
    >>> history.log(record)
    >>> # Later:
    >>> print(len(history))  # number of stored records
    >>> first = history[0]  # access first logged iteration

    Parameters
    ----------
    once_in_iterations : int, optional
        The recording frequency. A value of `n` means a record is stored every
        `n` iterations (e.g., `n=3` stores iterations 0, 3, 6, ...).
        Defaults to 1 (record every iteration).

    Attributes
    ----------
    once_in_iterations : int
        The configured recording frequency.
    _counter : int
        Internal counter tracking the total number of `log()` calls
        (i.e., total iterations processed, not just recorded ones).
    _logs : list[IterationRecord]
        List of stored iteration records. Only iterations matching the
        recording frequency are appended.

    Raises
    ------
    ValueError
        If `once_in_iterations` is not a positive integer.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        log
        reset
    """

    def __init__(self, once_in_iterations: int = 1) -> None:
        """
        Initialize a new PipelineLogger instance.
        """
        if once_in_iterations < 1:
            raise ValueError("once_in_iterations must be a positive integer")

        self._logs: list[IterationRecord] = []
        self._counter: int = 0
        self.once_in_iterations = once_in_iterations

    def log(self, record: IterationRecord) -> None:
        """Store an iteration record based on the configured frequency.

        The record is stored only if the current internal counter is divisible
        by `once_in_iterations`. The internal counter increments on every call,
        regardless of whether the record is stored.

        Parameters
        ----------
        record : IterationRecord
            The iteration snapshot to potentially store. The `record.iteration`
            should ideally match the logger's internal state, though this is
            not enforced.
        """
        if self._counter % self.once_in_iterations == 0:
            self._logs.append(record)
        self._counter += 1

    def reset(self) -> None:
        """Clear all stored records and reset the internal iteration counter.

        The recording frequency (`once_in_iterations`) is reset to 1
        to ensure a clean state for new runs.
        """
        self._logs.clear()
        self._counter = 0
        self.once_in_iterations = 1

    def __len__(self) -> int:
        """Return the number of stored iteration records.

        Enables use of `len(history)` to get the count of recorded iterations
        (not total iterations run).

        Returns
        -------
        int
            Number of `IterationRecord` objects currently stored.
        """
        return len(self._logs)

    def __getitem__(self, index: int) -> IterationRecord:
        """Access a stored iteration record by index.

        Supports both positive (0-based) and negative indexing (e.g., `-1` for last).
        Enables syntax like `history[0]` or `history[-1]`.

        Parameters
        ----------
        index : int
            Index of the desired record.

        Returns
        -------
        IterationRecord
            The recorded state of the specified iteration.

        Raises
        ------
        IndexError
            If the index is out of range for the current number of stored records.
        """
        if index >= len(self) or index < -len(self):
            raise IndexError(f"Index {index} out of range for container containing {len(self)} elements")
        return self._logs[index]
