""" The script implements the first step of the experiment """

from pathlib import Path

import numpy as np

from experimental_env.preparation.dataset_generator import (
    ConcreteDatasetGenerator,
    RandomDatasetGenerator,
)
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp

WORKING_DIR = Path("/home/danil/PycharmProjects/Projects/EM-algo-DT/experiment/stage_1")
SAMPLES_SIZE = 1000

np.random.seed(42)

r_generator = RandomDatasetGenerator()
mixtures = [
    [ExponentialModel],
    [GaussianModel],
    [WeibullModelExp],
    [WeibullModelExp, GaussianModel],
    [ExponentialModel, GaussianModel],
    [WeibullModelExp, WeibullModelExp],
    [ExponentialModel, ExponentialModel],
]
for models in mixtures:
    r_generator.generate(SAMPLES_SIZE, models, Path(WORKING_DIR), exp_count=100)

c_generator2 = ConcreteDatasetGenerator()
models = [ExponentialModel]
c_generator2.add_distribution(models[0], [1.0], 1.0)
c_generator2.generate(SAMPLES_SIZE, Path(WORKING_DIR), 5)

c_generator3 = ConcreteDatasetGenerator()
models = [GaussianModel]
c_generator3.add_distribution(models[0], [0, 1.0], 1.0)
c_generator3.generate(SAMPLES_SIZE, Path(WORKING_DIR), 5)

c_generator4 = ConcreteDatasetGenerator()
models = [WeibullModelExp]
c_generator4.add_distribution(models[0], [1.0, 1.0], 1.0)
c_generator4.generate(SAMPLES_SIZE, Path(WORKING_DIR), 5)

c_generator5 = ConcreteDatasetGenerator()
models = [WeibullModelExp]
c_generator5.add_distribution(models[0], [1.0, 1.0], 1.0)
c_generator5.generate(SAMPLES_SIZE, Path(WORKING_DIR), 5)

c_generator6 = ConcreteDatasetGenerator()
models = [WeibullModelExp]
c_generator6.add_distribution(models[0], [1.0, 0.5], 1.0)
c_generator6.generate(SAMPLES_SIZE, Path(WORKING_DIR), 5)

c_generator7 = ConcreteDatasetGenerator()
models = [GaussianModel, GaussianModel]
c_generator7.add_distribution(models[0], [-1.0, 2.5], 0.3)
c_generator7.add_distribution(models[1], [1.0, 0.5], 0.7)
c_generator7.generate(SAMPLES_SIZE, Path(WORKING_DIR), 10)

c_generator8 = ConcreteDatasetGenerator()
models = [GaussianModel, GaussianModel]
c_generator8.add_distribution(models[0], [0.0, 1.5], 0.6)
c_generator8.add_distribution(models[1], [1.0, 1.0], 0.4)
c_generator8.generate(SAMPLES_SIZE, Path(WORKING_DIR), 10)
