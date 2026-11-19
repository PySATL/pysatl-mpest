"""Mathematical assertions utilities for testing."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any

import numpy as np


def assert_is_scalar_type(val: Any) -> None:
    """Asserts that the provided value is a numpy scalar of type float64."""

    assert np.isscalar(val), f"Expected scalar, got {type(val)}"
    assert isinstance(val, np.float64), f"Expected np.float64, got {type(val)}"


def assert_is_array_type(val: Any, expected_shape: tuple[int, ...]) -> None:
    """Asserts that the provided value is a numpy array of type float64 with the expected shape."""

    assert isinstance(val, np.ndarray), f"Expected np.ndarray, got {type(val)}"
    assert val.dtype == np.float64, f"Expected float64 dtype, got {val.dtype}"
    assert val.shape == expected_shape, f"Expected shape {expected_shape}, got {val.shape}"
