"""Integration tests for the SciPy Powell optimizer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_mpest.optimizers.scipy_powell import ScipyPowell

from tests.helpers.objectives import bounded_objective


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_powell_finds_minimum_with_infinite_boundary() -> None:
    result = ScipyPowell().minimize(bounded_objective, [0.5, 0.05])

    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(1.0, abs=1e-3)
