"""Unit tests for ``IdentityMatcher``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
import pytest
from pysatl_mpest.initializers.cluster.matchers.identity_matcher import IdentityMatcher
from pysatl_mpest.initializers.cluster.table import ComponentTable
from pysatl_mpest.typings import UnivariateFloatArray


class SizeOnlyTable:
    """Component table stub that allows reading only the number of families."""

    def __init__(self, k: int) -> None:
        self.k = k

    def groups(self) -> tuple[tuple[int, ...], ...]:
        raise AssertionError("The matcher must read only k.")

    def cluster_means(self) -> UnivariateFloatArray:
        raise AssertionError("The matcher must read only k.")

    def loglikelihood(self, j: int, c: int) -> float:
        raise AssertionError("The matcher must read only k.")

    def failure(self, j: int, c: int) -> str | None:
        raise AssertionError("The matcher must read only k.")


@pytest.mark.parametrize("k", [1, 3], ids=["one-family", "three-families"])
def test_returns_identity_assignment(k: int) -> None:
    matcher = IdentityMatcher()
    table = cast(ComponentTable, SizeOnlyTable(k))

    result = matcher.match(table)

    assert result.ndim == 1
    assert result.dtype.kind == "i"
    np.testing.assert_array_equal(result, np.arange(k))


def test_reads_only_the_number_of_families() -> None:
    matcher = IdentityMatcher()
    table = cast(ComponentTable, SizeOnlyTable(3))

    result = matcher.match(table)

    np.testing.assert_array_equal(result, [0, 1, 2])
