"""
Internal module with numerical constants shared by `estimators` and `initializers`.

`NUMERICAL_TOLERANCE` is an absolute threshold at or below which M-step strategies treat a quantity
as zero in their closed-form updates and leave the parameter unchanged. Modules import the
name itself so that tests can replace the tolerance on a single module.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


NUMERICAL_TOLERANCE = 1e-9
