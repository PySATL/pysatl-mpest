"""Pure functions matching clusters to distribution families.

These functions know nothing about the component table or where the costs come from;
they only solve the assignment problem and canonicalize the order of identical families.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..typings import MultivariateFloatArray, UnivariateFloatArray, UnivariateIntArray


def solve_assignment(cost: MultivariateFloatArray) -> UnivariateIntArray | None:
    """Solve the assignment problem for a square cost matrix.

    Each row (family) is assigned a distinct column (cluster) so that the total cost is
    minimal. A cost of ``+inf`` marks a forbidden pair: the pair is never chosen when an
    assignment without forbidden pairs exists.

    Returns ``None`` when every assignment contains a forbidden pair. The function does
    not raise in this case because the caller, which knows why pairs are impossible,
    builds the explanatory message itself.

    Parameters
    ----------
    cost : MultivariateFloatArray
        Square ``k x k`` matrix of costs; ``+inf`` entries forbid the pair.

    Returns
    -------
    UnivariateIntArray or None
        Column assigned to each row, or ``None`` if no feasible assignment exists.

    Raises
    ------
    ValueError
        If ``cost`` contains ``NaN`` or ``-inf``. These values are a bug in the calling
        code, not a property of the data, so they are rejected before scipy sees them.
    """

    if np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("The cost matrix must not contain NaN or -inf.")

    try:
        _, columns = linear_sum_assignment(cost)
    except ValueError:
        return None

    return columns


def order_identical(
    assignment: UnivariateIntArray,
    groups: tuple[tuple[int, ...], ...],
    means: UnivariateFloatArray,
) -> UnivariateIntArray:
    """Order clusters of identical families by cluster mean.

    Within each group of identical families the clusters they received are redistributed
    among the group members: members are taken in the order they appear in the group and
    receive clusters sorted by the pair ``(cluster mean, column index)``, so the family
    with the smaller index gets the cluster with the smaller mean, and ties are broken by
    the column number. The mixture is unchanged by this reordering because identical
    families are interchangeable.

    Parameters
    ----------
    assignment : UnivariateIntArray
        Permutation where ``assignment[j]`` is the cluster assigned to family ``j``.
    groups : tuple of tuple of int
        Groups of identical family indices; every index belongs to exactly one group,
        indices inside a group are in ascending order.
    means : UnivariateFloatArray
        ``means[c]`` is the mean of cluster ``c``.

    Returns
    -------
    UnivariateIntArray
        New array of the same length; ``assignment`` is not modified.
    """

    result = assignment.copy()

    for members in groups:
        clusters = sorted((assignment[member] for member in members), key=lambda c: (means[c], c))
        for member, cluster in zip(members, clusters):
            result[member] = cluster

    return result
