"""Tests of the conversion of cluster labels to a partition matrix."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.typing import ArrayLike
from pysatl_mpest.initializers.cluster._table._validation import validate_partition
from pysatl_mpest.initializers.cluster.partitioners._labels import one_hot


@pytest.mark.parametrize(
    "labels",
    [np.array([0, 2, 1, 0], dtype=np.int32), np.array([0, 2, 1, 0], dtype=np.int64), [0, 2, 1, 0]],
    ids=["int32", "int64", "list"],
)
def test_puts_the_one_in_the_column_of_the_label(labels: ArrayLike) -> None:
    H = one_hot(labels, 3)

    assert H.dtype == np.float64
    np.testing.assert_array_equal(H, [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])


@pytest.mark.parametrize("labels", [[0, 2], [0, -1]], ids=["too-large", "negative"])
def test_label_outside_the_range_gives_a_row_without_a_one(labels: list[int]) -> None:
    H = one_hot(labels, 2)

    np.testing.assert_array_equal(H, [[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="^Every row of a partition matrix must contain exactly one 1"):
        validate_partition(H, n=2, k=2)


def test_number_without_labels_gives_a_zero_column() -> None:
    H = one_hot([0, 0], 2)

    np.testing.assert_array_equal(H, [[1, 0], [1, 0]])
