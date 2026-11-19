"""Iteration history recording for iterative estimation pipelines."""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

from ...core import MixtureModel
from ...typings import MultivariateFloatArray, UnivariateFloatArray
from .pruner import Pruner


@dataclass
class IterationRecord:
    """Snapshot of a single pipeline iteration after pruning.

    Parameters
    ----------
    iteration : int
        Zero-based iteration index.
    mixture : MixtureModel
        State of the mixture model after pruning in this iteration.
    X : UnivariateFloatArray
        Input data sample.
    H : MultivariateFloatArray | None
        Responsibility matrix (posterior probabilities) if computed.
    pruners_used : list[Pruner] | None
        Pruners executed during this iteration, or None if none were applied.
    error : Exception | None
        Exception raised by a step. The failed iteration has no record of its own unless it is the first
        one, so an error from a later iteration is set on the last recorded one.
    """

    iteration: int
    mixture: MixtureModel
    X: UnivariateFloatArray
    H: MultivariateFloatArray | None
    pruners_used: list[Pruner] | None
    error: Exception | None


class IterationsHistory:
    """Container for recorded pipeline iteration snapshots.

    Stores an :class:`IterationRecord` every ``once_in_iterations`` steps.
    The *i*-th element corresponds to iteration `i * once_in_iterations`.

    Parameters
    ----------
    once_in_iterations : int, optional
        Recording frequency: store a snapshot every *n* iterations. Defaults to 1.

    Attributes
    ----------
    once_in_iterations : int
        Configured recording frequency.

    Raises
    ------
    ValueError
        If `once_in_iterations` is not a positive integer.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        save_record
        reset
    """

    def __init__(self, once_in_iterations: int = 1) -> None:
        if once_in_iterations < 1:
            raise ValueError("Parameter once_in_iterations must be a positive integer")

        self._history: list[IterationRecord] = []
        self._counter: int = 0
        self.once_in_iterations = once_in_iterations

    def save_record(self, record: IterationRecord) -> None:
        """Store an iteration record if the iteration counter matches the recording frequency."""

        if self._counter % self.once_in_iterations == 0:
            self._history.append(record)
        self._counter += 1

    def reset(self) -> None:
        """Clear recorded history, reset the counter and set the recording frequency back to 1."""

        self._history.clear()
        self._counter = 0
        self.once_in_iterations = 1

    def __len__(self) -> int:
        """Number of recorded iteration snapshots."""

        return len(self._history)

    def __getitem__(self, index: int) -> IterationRecord:
        """Access recorded iteration snapshot by index (supports negative indexing)."""

        if index >= len(self) or index < -len(self):
            raise IndexError(f"Index {index} out of range for container containing {len(self)} elements")
        return self._history[index]
