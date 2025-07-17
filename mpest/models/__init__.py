"""Module which represents models and abstract classes for extending"""

from mpest.models.abstract_model import (
    AModel,
    AModelDifferentiable,
    AModelWithGenerator,
)
from mpest.models.exponential import ExponentialModel
from mpest.models.gaussian import GaussianModel
from mpest.models.uniform import Uniform
from mpest.models.weibull import WeibullModelExp
from mpest.models.cauchy import Cauchy
from mpest.models.pareto import Pareto
from mpest.models.beta import Beta

ALL_MODELS: dict[str, type[AModel]] = {
    GaussianModel().name: GaussianModel,
    WeibullModelExp().name: WeibullModelExp,
    ExponentialModel().name: ExponentialModel,
    Cauchy().name: Cauchy,
    Pareto().name: Pareto,
    Beta().name: Beta,
    Uniform().name: Uniform,
}
