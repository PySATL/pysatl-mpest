""" The script implements the second step of the experiment """

import random
from pathlib import Path

from experimental_env.experiment.estimators import (
    LikelihoodEstimator,
    LMomentsEstimator,
)
from experimental_env.experiment.experiment_executors.random_executor import (
    RandomExperimentExecutor,
)
from experimental_env.preparation.dataset_parser import SamplesDatasetParser
from mpest.em.breakpointers import StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)

SOURCE_DIR = Path("/home/danil/PycharmProjects/Projects/EM-algo-DT/experiment/stage_1")
WORKING_DIR = Path("/home/danil/PycharmProjects/Projects/EM-algo-DT/experiment/stage_2")

random.seed(42)

# Parse stage 1
parser = SamplesDatasetParser()
datasets = parser.parse(SOURCE_DIR)

# Execute stage 2
executor = RandomExperimentExecutor(WORKING_DIR, 5)
executor.execute(
    datasets,
    LMomentsEstimator(
        StepCountBreakpointer(max_step=16),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
    ),
)

executor = RandomExperimentExecutor(WORKING_DIR, 5)
executor.execute(
    datasets,
    LikelihoodEstimator(
        StepCountBreakpointer(max_step=16),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
    ),
)
