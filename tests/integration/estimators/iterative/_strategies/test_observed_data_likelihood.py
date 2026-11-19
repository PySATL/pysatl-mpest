"""Tests for the observed data likelihood strategy."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pysatl_mpest.estimators.iterative._strategies import observed_data_likelihood_strategy
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.families import family
from tests.helpers.pipeline import make_state


def odl_block(component_id: int = 1, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


@pytest.fixture
def sample_mixture() -> tuple[MixtureModel, np.ndarray]:
    mixture = MixtureModel([normal(0.0, 1.0), normal(5.0, 1.0)], weights=[0.5, 0.5])
    return mixture, np.asarray(mixture.generate(size=4000, random_state=1))


def test_recovers_the_parameters_in_presence_of_an_unsupported_point() -> None:
    truth = MixtureModel([family(FamilyName.EXPONENTIAL)(lambda_=0.5), family(FamilyName.EXPONENTIAL)(lambda_=1.5)])
    sample = np.asarray(truth.generate(size=1000, random_state=42))
    start = family(FamilyName.EXPONENTIAL)(lambda_=3.0)
    mixture = MixtureModel([truth[0], start])
    _, clean = observed_data_likelihood_strategy(
        start, make_state(sample, None, mixture), odl_block(), ScipyNelderMead()
    )
    _, contaminated = observed_data_likelihood_strategy(
        start, make_state(np.concatenate((sample, [-0.5])), None, mixture), odl_block(), ScipyNelderMead()
    )

    assert contaminated["lambda_"] == pytest.approx(clean["lambda_"], rel=1e-4)
    assert contaminated["lambda_"] != pytest.approx(3.0)


def test_uniform_component_extends_its_support_over_points_outside_it() -> None:
    sample = np.random.default_rng(42).uniform(0.0, 10.0, size=1000)
    component = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=4.0, upper_bound=6.0)
    mixture = MixtureModel([component])

    _, fitted = observed_data_likelihood_strategy(
        component, make_state(sample, None, mixture), odl_block(component_id=0), ScipyNelderMead()
    )

    assert fitted["lower_bound"] <= sample.min()
    assert fitted["upper_bound"] >= sample.max()


def test_recovers_the_parameters_of_a_component_from_a_sample(sample_mixture: tuple[MixtureModel, np.ndarray]) -> None:
    mixture, sample = sample_mixture
    start = normal(4.0, 2.0)

    _, new_params = observed_data_likelihood_strategy(
        start, make_state(sample, None, mixture), odl_block(), ScipyNelderMead()
    )

    assert new_params["mu"] == pytest.approx(5.0, abs=0.15)
    assert new_params["sigma"] == pytest.approx(1.0, abs=0.1)


def test_recovers_the_scale_of_a_component_with_fixed_mean(sample_mixture: tuple[MixtureModel, np.ndarray]) -> None:
    mixture, sample = sample_mixture
    start = family(FamilyName.NORMAL).view(mu=5.0)(sigma=2.0)

    _, new_params = observed_data_likelihood_strategy(
        start, make_state(sample, None, mixture), odl_block(), ScipyNelderMead()
    )

    assert set(new_params) == {"sigma"}
    assert new_params["sigma"] == pytest.approx(1.0, abs=0.1)
