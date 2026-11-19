"""Shared data and helpers for estimator integration tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_mpest.core import MixtureModel

from tests.helpers.estimator_data import TRUE_WEIGHTS, normal, sample_of


@pytest.fixture
def true_mixture() -> MixtureModel:
    return MixtureModel([normal(0.0, 1.0), normal(6.0, 1.5)], weights=TRUE_WEIGHTS)


@pytest.fixture
def X(true_mixture: MixtureModel) -> np.ndarray:
    return sample_of(true_mixture)


@pytest.fixture
def start() -> MixtureModel:
    return MixtureModel([normal(1.0, 2.0), normal(4.0, 2.0)], weights=[0.5, 0.5])
