"""Shared test doubles of initializers."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence

import numpy as np
from pysatl_core.families import ParametricFamily
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.initializers.initializer import Initializer
from pysatl_mpest.typings import UnivariateFloatArray

type Build = Callable[[UnivariateFloatArray, Sequence[ParametricFamily], np.random.Generator], MixtureModel]


class ScriptedInitializer(Initializer):
    """Initializer whose result is scripted by the test."""

    def __init__(self, build: Build) -> None:
        self._build = build
        self.calls: list[tuple[UnivariateFloatArray, Sequence[ParametricFamily], np.random.Generator]] = []

    def _initialize(
        self, X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
    ) -> MixtureModel:
        self.calls.append((X, families, rng))
        return self._build(X, families, rng)
