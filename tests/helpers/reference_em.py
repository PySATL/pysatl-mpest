"""Independent reference implementation of the EM algorithm for normal mixtures."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import NDArray
from pysatl_mpest.core import MixtureModel
from scipy import stats
from scipy.special import logsumexp

from tests.helpers.estimator_data import parameters


def reference_em(
    X: NDArray[np.float64],
    mixture: MixtureModel,
    iterations: int,
    fixed_mu: tuple[bool, ...] = (),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The EM algorithm for a mixture of normals written directly in numpy.

    Parameters
    ----------
    X : NDArray[np.float64]
        Input data sample.
    mixture : MixtureModel
        Initial mixture model to iterate from.
    iterations : int
        Number of EM iterations to perform.
    fixed_mu : tuple[bool, ...], optional
        Marks the components whose mean is not updated; their standard deviation is measured
        around the fixed mean. Defaults to no fixed means.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        The means, standard deviations and weights after ``iterations`` iterations.
    """

    params = parameters(mixture)
    mu = np.array([p["mu"] for p in params])
    sigma = np.array([p["sigma"] for p in params])
    weights = np.array(mixture.weights)
    frozen = np.array(fixed_mu or (False,) * len(mu))

    for _ in range(iterations):
        joint = np.log(weights) + stats.norm(mu, sigma).logpdf(X[:, None])
        resp = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        total = resp.sum(axis=0)
        weights = total / len(X)
        mu = np.where(frozen, mu, resp.T @ X / total)
        sigma = np.sqrt((resp * (X[:, None] - mu) ** 2).sum(axis=0) / total)

    return mu, sigma, weights
