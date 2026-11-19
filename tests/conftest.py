"""Pytest configuration and shared fixtures for the PySATL project tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random

import numpy as np
import pytest
from hypothesis import settings

# Fix seed for Hypothesis to ensure deterministic parameter generation
settings.register_profile("deterministic", derandomize=True)
settings.load_profile("deterministic")


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Sets random seeds for reproducible tests across the suite."""
    random.seed(42)
    np.random.seed(42)
