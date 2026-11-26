"""Pytest configuration and shared fixtures for the PySATL project tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import os
import random

import numpy as np
import pytest
from hypothesis import settings

# mutmut forks the test process, and the OpenMP thread pool of scikit-learn hangs in a forked child.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Fix seed for Hypothesis to ensure deterministic parameter generation
settings.register_profile("deterministic", derandomize=True)
settings.load_profile("deterministic")


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Sets random seeds for reproducible tests across the suite."""
    random.seed(42)
    np.random.seed(42)
