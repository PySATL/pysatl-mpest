"""
Pysatl-mpest: A Python package for statistical analysis using mixture models.

This package provides tools for creating, manipulating, and estimating parameters
of finite mixture models of continuous probability distributions.

Mixture components are `pysatl-core` distributions. They are created from the families
registered in `pysatl-core`, for example
``ParametricFamilyRegister.get(FamilyName.NORMAL)(mu=0.0, sigma=1.0)``, and parameters
can be fixed with ``family.view(...)``. Importing this package registers the built-in
families of `pysatl-core`.

Key modules include:
- `core`: Provides the main `MixtureModel` class.
- `estimators`: Provides the EM, ECM and ECME estimators and the iterative pipeline they are built on.
- `optimizers`: Provides the numerical optimizers used by the maximization step.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.families import configure_families_register

configure_families_register()
