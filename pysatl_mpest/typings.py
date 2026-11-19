"""Type aliases for array shapes and scalar values."""

__author__ = "Aleksandra Ri, Viktor Khanukaev, Totmyanin Danil"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

Scalar = int | float | np.floating | np.integer
BoolScalar = bool | np.bool_

type UnivariateFloatArray = np.ndarray[tuple[int], np.dtype[np.float64]]
type MultivariateFloatArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

type UnivariateIntArray = np.ndarray[tuple[int], np.dtype[np.integer]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
