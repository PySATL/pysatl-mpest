"""
Helpers for obtaining real pysatl-core parametric families in tests.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.families import ParametricFamily, ParametricFamilyRegister, configure_families_register
from pysatl_core.types import FamilyName


def family(name: FamilyName) -> ParametricFamily:
    """
    Return a registered pysatl-core family, registering the built-in ones if needed.

    Parameters
    ----------
    name : FamilyName
        Name of the family to look up.

    Returns
    -------
    ParametricFamily
        The registered family.
    """
    configure_families_register()
    return ParametricFamilyRegister.get(name)
