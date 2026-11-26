"""
Checks of a mixture against a sample: which points the mixture does not cover.

The functions here only report what they find; the caller chooses the exception and its message.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import NamedTuple

import numpy as np

from ..typings import UnivariateFloatArray, UnivariateIntArray
from .mixture import MixtureModel


class SampleCoverage(NamedTuple):
    """Sample points where the log-density of the mixture is not finite.

    Indices are in ascending order in both arrays, and the arrays do not intersect.

    Attributes
    ----------
    uncovered : UnivariateIntArray
        Indices of points where the log-density is ``-inf``: no component covers the point.
    undefined : UnivariateIntArray
        Indices of points where the log-density is ``NaN`` or ``+inf``: the density is undefined.
    """

    uncovered: UnivariateIntArray
    undefined: UnivariateIntArray


def sample_coverage(mixture: MixtureModel, X: UnivariateFloatArray) -> SampleCoverage:
    """Find the sample points where the log-density of the mixture is not finite.

    A point is *uncovered* when the log-density of the mixture there is ``-inf``, and *undefined*
    when it is ``NaN`` or ``+inf`` — for example when a component has a ``NaN`` parameter or its
    density has a pole at the point. The criterion is the log-density of the mixture with its
    weights, so a component whose log-weight is ``-inf`` covers nothing. The function only
    reports; it raises nothing itself.

    Parameters
    ----------
    mixture : MixtureModel
        The mixture to check.
    X : UnivariateFloatArray
        A one-dimensional sample; its dimensionality and finiteness are the caller's
        responsibility and are not checked here.

    Returns
    -------
    SampleCoverage
        Indices of the uncovered and undefined points; both arrays are empty when the
        log-density is finite at every point.
    """

    log_density = mixture.lpdf(X)

    uncovered = np.flatnonzero(log_density == -np.inf)
    undefined = np.flatnonzero(np.isnan(log_density) | np.isposinf(log_density))

    return SampleCoverage(uncovered=uncovered, undefined=undefined)
