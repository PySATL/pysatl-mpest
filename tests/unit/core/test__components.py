"""
Unit tests of the data types that the component helpers exchange with a pysatl-core distribution.

A stub stands in for the distribution, so the tests can choose what it receives and what it returns.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import CharacteristicName
from pysatl_mpest.core import _components as cc

from tests.helpers.math_assertions import assert_is_array_type, assert_is_scalar_type


class RecordingComponent:
    """A stand-in for a distribution that records the data it gets and returns a prepared result."""

    def __init__(self, result: ArrayLike) -> None:
        self.result = result
        self.received_name: CharacteristicName | None = None
        self.received_data: np.ndarray | None = None

    def calculate_characteristic(self, name: CharacteristicName, X: np.ndarray) -> ArrayLike:
        self.received_name, self.received_data = name, X
        return self.result


def component_returning(result: ArrayLike) -> tuple[ParametricFamilyDistribution, RecordingComponent]:
    stub = RecordingComponent(result)
    return cast(ParametricFamilyDistribution, stub), stub


evaluation_cases = pytest.mark.parametrize(
    ("helper", "name"),
    [(cc.lpdf, CharacteristicName.LPDF), (cc.pdf, CharacteristicName.PDF)],
    ids=["lpdf", "pdf"],
)


@evaluation_cases
def test_asks_for_the_right_characteristic(helper, name) -> None:
    component, stub = component_returning(np.zeros(2))

    helper(component, [1.0, 2.0])

    assert stub.received_name == name


@evaluation_cases
@pytest.mark.parametrize(
    "data",
    [[1, 2], np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.float32)],
    ids=["list", "int32", "float32"],
)
def test_gives_float64_data_to_the_component(helper, name, data) -> None:
    component, stub = component_returning(np.zeros(2))

    helper(component, data)

    assert stub.received_data is not None
    assert stub.received_data.dtype == np.float64


@evaluation_cases
@pytest.mark.parametrize("dtype", [np.int32, np.float32], ids=["int32", "float32"])
def test_returns_float64_whatever_the_component_returns(helper, name, dtype) -> None:
    component, _ = component_returning(np.array([1, 2], dtype=dtype))

    assert_is_array_type(helper(component, [1.0, 2.0]), (2,))


@evaluation_cases
def test_returns_a_float64_scalar_for_a_scalar_result(helper, name) -> None:
    component, _ = component_returning(np.float32(0.5))

    assert_is_scalar_type(helper(component, 1.0))


@evaluation_cases
def test_returns_an_array_for_a_one_element_array(helper, name) -> None:
    component, _ = component_returning(np.array([0.5]))

    assert_is_array_type(helper(component, [1.0]), (1,))
