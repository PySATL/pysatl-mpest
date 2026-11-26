"""
Initializers build a starting :class:`~pysatl_mpest.core.MixtureModel` from a sample and a list of component families.

The package is under development and exports only its errors for now.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .exceptions import ComponentEstimationError, InitializationError

__all__ = ["ComponentEstimationError", "InitializationError"]
