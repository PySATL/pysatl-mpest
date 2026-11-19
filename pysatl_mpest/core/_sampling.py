"""Sampling strategy that honors ``random_state``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.strategies import SamplingStrategy
from pysatl_core.types import CharacteristicName, NumericArray


class DeterministicSampling(SamplingStrategy):
    """Inverse transform sampling that honors ``random_state``.

    Draws uniform variates and maps them through the PPF. The default sampling strategy of `pysatl-core`
    ignores ``random_state``, which makes the samples irreproducible.

    The ``random_state`` option is a seed or a `numpy.random.Generator`. A generator is used in place,
    so its state advances; this lets several calls share one random stream.
    """

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
        seed = options.get("random_state")
        rng = np.random.default_rng(seed)

        return distr.calculate_characteristic(CharacteristicName.PPF, rng.random(n))
