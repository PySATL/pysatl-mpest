"""Module which describes configuration of experiments"""

import os

from mpest.optimizers import ScipyCG, ScipyNewtonCG, ScipySLSQP, ScipyTNC
from scripts.shared import EXAMPLES

CPU_COUNT = os.cpu_count()
MAX_WORKERS_PERCENT = 0.75
MAX_WORKERS = min(round(CPU_COUNT * MAX_WORKERS_PERCENT), CPU_COUNT) if CPU_COUNT is not None else 1
# MAX_WORKERS = 4

RESULTS_FOLDER = EXAMPLES / "results"

TESTS_OPTIMIZERS = [
    ScipyCG(),
    ScipyNewtonCG(),
    ScipySLSQP(),
    ScipyTNC(),
]
