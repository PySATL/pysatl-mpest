"""
Integration test for the side effect of importing the package.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import subprocess
import sys


def test_import_registers_builtin_families() -> None:
    # A subprocess is used because the test session has already registered the families.

    script = (
        "import pysatl_mpest\n"
        "from pysatl_core.families import ParametricFamilyRegister\n"
        "from pysatl_core.types import FamilyName\n"
        "for name in (FamilyName.NORMAL, FamilyName.EXPONENTIAL, FamilyName.CONTINUOUS_UNIFORM):\n"
        "    ParametricFamilyRegister.get(name)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
