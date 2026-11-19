"""Integration tests for the SciPy Nelder-Mead optimizer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_mpest.optimizers.scipy_nelder_mead import ScipyNelderMead

from tests.helpers.objectives import bounded_objective


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_nelder_mead_finds_minimum_with_infinite_boundary() -> None:
    result = ScipyNelderMead().minimize(bounded_objective, [0.5, 0.05])

    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(1.0, abs=1e-3)
