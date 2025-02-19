"""Module which represents models and abstract classes for extending"""

from mpest.models.abstract_model import (
    AModel,
    AModelDifferentiable,
    AModelWithGenerator,
)
from mpest.models.exponential import ExponentialModel
from mpest.models.gaussian import GaussianModel
from mpest.models.weibull import WeibullModelExp

ALL_MODELS: dict[str, type[AModel]] = {
    GaussianModel().name: GaussianModel,
    WeibullModelExp().name: WeibullModelExp,
    ExponentialModel().name: ExponentialModel,
}
