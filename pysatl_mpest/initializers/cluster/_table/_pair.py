"""Evaluation of a pair of a family and a cluster.

A component estimator only computes numbers. This module checks its answer, builds the component and
computes the log-likelihood of the cluster under it. The checks live here rather than in estimators so
that they are the same for every estimator, including a user-provided one.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Any

import numpy as np
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.types import FamilyName

from ....core import _components
from ....typings import UnivariateFloatArray
from ...exceptions import ComponentEstimationError
from ..component_estimators.base import ComponentEstimator
from . import _validation

_QUARTILES = (0.25, 0.75)


@dataclass(frozen=True, slots=True)
class PairEvaluation:
    """Result of evaluating a pair of a family and a cluster.

    An estimated pair has a component, a finite log-likelihood and no reason. A pair that does not
    cover its cluster has a component and a non-finite log-likelihood as computed, with a reason. A
    pair that is not estimated has no component, log-likelihood ``-inf`` and a reason.

    Attributes
    ----------
    component : ParametricFamilyDistribution or None
        Component built from the values returned by the estimator, or ``None`` when it was not estimated.
    loglikelihood : float
        Log-likelihood of the points of the cluster under the component. It is not normalized: ``NaN``
        and ``+inf`` are kept as computed, and only a pair without a component always has ``-inf``.
    reason : str or None
        Why the pair is not estimated or does not cover its cluster; ``None`` for an estimated pair.
    """

    component: ParametricFamilyDistribution | None
    loglikelihood: float
    reason: str | None


def evaluate_pair(
    X: UnivariateFloatArray,
    h: UnivariateFloatArray,
    family: ParametricFamily,
    component_estimator: ComponentEstimator,
    scale: float,
) -> PairEvaluation:
    """Estimate the component of one family on one cluster and evaluate the result.

    A :class:`~.ComponentEstimationError` raised by the estimator becomes the state "not estimated"
    with its message as the reason. Every other exception of the estimator passes through unchanged.

    Parameters
    ----------
    X : UnivariateFloatArray
        The whole sample.
    h : UnivariateFloatArray
        Column of the partition matrix as ``float64``: 1.0 for the points of the cluster and 0.0 for
        the others; the points of the cluster are ``X[h > 0]``.
    family : ParametricFamily
        The family of the component.
    component_estimator : ComponentEstimator
        Estimator of the free parameters of the family.
    scale : float
        Standard deviation of the whole sample; the caller computes it.

    Returns
    -------
    PairEvaluation
        The pair is estimated when the component passes all checks and the log-likelihood of the
        cluster is finite. It does not cover the cluster when the component exists but that
        log-likelihood is not finite. It is not estimated when the estimator declines the cluster or
        a check before the log-likelihood fails.

    Raises
    ------
    TypeError
        If the estimator returned keys other than the free parameters of the family or a value that
        is not a real scalar. Both are errors in the estimator, not properties of the data.

    Notes
    -----
    The checks are performed in this order:

    #. enough distinct values in the cluster, before the estimator is called;
    #. keys of the answer of the estimator;
    #. real scalar type and finiteness of every value;
    #. constraints of the parametrization;
    #. interquartile width of the component relative to ``scale``;
    #. the temporary absolute M-step tolerance;
    #. finiteness of the cluster log-likelihood and points outside the support.

    The whole evaluation runs with numpy warnings suppressed: a numerical failure is reported by the
    state of the pair instead of a warning.
    """

    with np.errstate(all="ignore"):
        points = X[h > 0]

        names = _components.family_free_parameter_names(family)
        n_distinct = len(np.unique(points))
        if not _validation.has_enough_distinct_values(n_distinct, len(names)):
            return PairEvaluation(
                None,
                -np.inf,
                f"the cluster has {n_distinct} distinct values, "
                f"fewer than the {len(names)} free parameters of the family.",
            )

        try:
            raw: dict[str, Any] = component_estimator.estimate(X, h, family)
        except ComponentEstimationError as error:
            return PairEvaluation(None, -np.inf, str(error))

        if set(raw) != set(names):
            raise TypeError(
                f"The component estimator returned parameters {sorted(raw)}, "
                f"but the free parameters of the family are {sorted(names)}."
            )

        prepared = _build_component(raw, names, family)
        if isinstance(prepared, str):
            return PairEvaluation(None, -np.inf, prepared)

        free, component = prepared
        if failure := _width_failure(component, scale) or _m_step_failure(family, free):
            return PairEvaluation(None, -np.inf, failure)

        logdensities = _components.lpdf(component, points)
        loglikelihood = float(np.sum(logdensities))
        if np.isfinite(loglikelihood):
            return PairEvaluation(component, loglikelihood, None)

        outside = points[np.isneginf(logdensities)]
        if len(outside) > 0:
            examples = ", ".join(str(float(x)) for x in outside[:3])
            reason = (
                f"{len(outside)} of {len(points)} points of the cluster are outside the support "
                f"of the component, for example {examples}."
            )
        else:
            reason = f"the log-likelihood of the cluster is {loglikelihood}."

        return PairEvaluation(component, loglikelihood, reason)


def _build_component(
    raw: dict[str, Any],
    names: tuple[str, ...],
    family: ParametricFamily,
) -> tuple[dict[str, float], ParametricFamilyDistribution] | str:
    """Free parameters and the component built from the estimator answer, or the reason they cannot be built."""

    for name in names:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"The component estimator returned a value of type {type(value).__name__} "
                f"for parameter '{name}'; a real scalar is expected."
            )

    free: dict[str, float] = {name: float(raw[name]) for name in names}
    if not all(np.isfinite(value) for value in free.values()):
        return f"the estimated parameters are not finite: {free}."

    # `distribution` also takes optional keyword arguments of other types, so mypy accepts `**` only for `Any` values.
    parameters: dict[str, Any] = free
    try:
        component = family.distribution(**parameters)
    except ValueError as error:
        return f"the estimated parameters {free} violate the constraints of the parametrization: {error}"

    return free, component


def _width_failure(component: ParametricFamilyDistribution, scale: float) -> str | None:
    """Failure message for a component whose interquartile width is degenerate relative to ``scale``, if any."""

    lower, upper = np.asarray(_components.ppf(component, _QUARTILES))
    width = upper - lower
    if _validation.component_is_wide_enough(width, scale):
        return None

    return (
        f"the interquartile width of the component {width} does not exceed "
        f"{_validation.RELATIVE_WIDTH_TOLERANCE} of the sample standard deviation {scale}."
    )


def _m_step_failure(family: ParametricFamily, free: dict[str, float]) -> str | None:
    """Failure message for a component whose quantity the M-step would never update, if any.

    Temporary check until the M-step tolerance becomes relative (issue #51). Only parametrizations
    with an analytic M-step are checked, and a fixed parameter is not checked because the M-step does
    not update it anyway.
    """

    name, parametrization, _ = _components.family_key(family)

    if (name, parametrization) == (FamilyName.NORMAL, "meanStd") and "sigma" in free:
        value = free["sigma"] ** 2
        if not _validation.exceeds_m_step_tolerance(value):
            return (
                f"the variance of the component {value} does not exceed "
                f"the absolute M-step tolerance {_validation.NUMERICAL_TOLERANCE}; rescale the data."
            )

    if (name, parametrization) == (FamilyName.EXPONENTIAL, "rate"):
        value = 1 / free["lambda_"]
        if not _validation.exceeds_m_step_tolerance(value):
            return (
                f"the mean of the component {value} does not exceed "
                f"the absolute M-step tolerance {_validation.NUMERICAL_TOLERANCE}; rescale the data."
            )

    return None
