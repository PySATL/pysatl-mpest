"""Shared data and helpers for estimator integration tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators import ECM
from pysatl_mpest.estimators.iterative import LikelihoodBreakpointer, StepBreakpointer
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead

from tests.helpers.families import family

N_SAMPLES = 2000
TRUE_WEIGHTS = [0.4, 0.6]
PRUNING_THRESHOLD = 0.05


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    """Create a normal component with the given parameters."""
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def exponential(lambda_: float) -> ParametricFamilyDistribution:
    """Create an exponential component with the given rate."""
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


def parameters(mixture: MixtureModel) -> list[dict[str, float]]:
    """Collect all component parameters from the mixture."""
    return [cc.all_parameters(component) for component in mixture]


def sample_of(mixture: MixtureModel, seed: int = 0) -> np.ndarray:
    """Generate a reproducible sample from the mixture."""
    return np.asarray(mixture.generate(size=N_SAMPLES, random_state=seed))


def ecm_for(iterations: int, optimizer: Optimizer | None = None, tolerance: float | None = None) -> ECM:
    """An ECM that runs exactly ``iterations`` iterations, or fewer if the likelihood stops changing."""

    breakpointers = [StepBreakpointer(iterations)]
    if tolerance is not None:
        breakpointers.append(LikelihoodBreakpointer(tolerance))
    return ECM(breakpointers=breakpointers, pruners=[], optimizer=optimizer or ScipyNelderMead())


def fitted_arrays(mixture: MixtureModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component means, standard deviations, and mixture weights."""
    params = parameters(mixture)
    return (
        np.array([p["mu"] for p in params]),
        np.array([p["sigma"] for p in params]),
        np.array(mixture.weights),
    )


def stray_start(stray_positions: tuple[int, ...]) -> MixtureModel:
    """The two true clusters plus components with a negligible weight that are inserted at the given positions."""

    stray_weight = 0.01
    components = [normal(0.0, 1.0), normal(6.0, 1.0)]
    weights = [(1.0 - stray_weight * len(stray_positions)) / 2] * 2
    for position, mu in zip(stray_positions, (3.0, -3.0, 9.0)):
        components.insert(position, normal(mu, 1.0))
        weights.insert(position, stray_weight)
    return MixtureModel(components, weights=weights)
