"""Stopping strategy based on mixture log-likelihood convergence."""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import warnings

import numpy as np
from numpy.typing import NDArray

from ....typings import Scalar
from ..breakpointer import Breakpointer
from ..pipeline_state import PipelineState


class LikelihoodBreakpointer(Breakpointer):
    """Stops the pipeline when the log-likelihood of the mixture converges.

    This breakpointer terminates the iterative estimation process when the
    absolute difference between the current and previous log-likelihood values
    falls below a specified threshold:

    .. math::
        |L_{t+1} - L_t| < threshold

    Only observations whose mixture log-density is not ``-inf`` contribute to
    the sum. Consecutive sums are compared only when the same observations
    contribute; a changed support mask establishes a new comparison baseline.
    Thus, this sum can be finite even when ``mixture.loglikelihood(X)`` is
    ``-inf`` because that method includes observations outside the support.

    An empty support or a non-finite sum emits a ``RuntimeWarning`` and stops
    the pipeline. The stored sum and support mask are cleared only by
    :meth:`reset`, which :meth:`Pipeline.fit` calls before every fitting run.

    Parameters
    ----------
    threshold : Scalar
        The convergence threshold for the log-likelihood difference.
        Must be a positive number.

    Attributes
    ----------
    threshold : Scalar
        The convergence threshold.

    Raises
    ------
    ValueError
        If `threshold` is not greater than 0.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        check
        reset
    """

    threshold: Scalar

    def __init__(self, threshold: Scalar) -> None:
        self._validate(threshold)
        self.threshold = threshold
        self._likelihood_old: float | None = None
        self._support_mask_old: NDArray[np.bool_] | None = None

    def _validate(self, threshold: Scalar) -> None:
        if threshold <= 0.0:
            raise ValueError("The threshold must be greater than 0")

    def check(self, state: PipelineState) -> bool:
        """Check convergence of the log-likelihood over supported observations.

        See the class description for how the support mask and the baseline are handled.

        Parameters
        ----------
        state : PipelineState
            The current state of the pipeline, which must contain a valid
            `curr_mixture` and data `X`.

        Returns
        -------
        bool
            True when comparable sums differ by less than ``threshold``, or
            when the sum is non-finite or has no contributing observations.
            The first call and any change of support return False.

        Warns
        -----
        RuntimeWarning
            If no observation is inside the support or the sum is non-finite,
            including log-densities containing NaN or positive infinity.
        """
        lpdf = np.asarray(state.curr_mixture.lpdf(state.X), dtype=np.float64)
        # Zero-density observations have no E-step responsibilities and do not contribute here.
        support_mask = ~np.isneginf(lpdf)
        in_support = lpdf[support_mask]
        likelihood_new = float(np.sum(in_support))

        if in_support.size == 0 or not np.isfinite(likelihood_new):
            warnings.warn(
                "The log-likelihood cannot be evaluated: no data point is inside the support of the mixture "
                "or the log-density is not finite. Stopping the pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return True

        if (
            self._likelihood_old is None
            or self._support_mask_old is None
            or not np.array_equal(support_mask, self._support_mask_old)
        ):
            self._likelihood_old = likelihood_new
            self._support_mask_old = support_mask.copy()
            return False

        if abs(likelihood_new - self._likelihood_old) < self.threshold:
            return True

        self._likelihood_old = likelihood_new
        return False

    def reset(self) -> None:
        """Clear the stored likelihood baseline and support mask."""
        self._likelihood_old = None
        self._support_mask_old = None
