"""Module wich contains supported distributions"""

from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp

Model = GaussianModel | WeibullModelExp | ExponentialModel
