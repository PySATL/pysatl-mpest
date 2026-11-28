"""End-to-end scenarios: the examples of the task section of the initializers design, run as written."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import CharacteristicName, FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import EM
from pysatl_mpest.estimators.iterative import StepBreakpointer
from pysatl_mpest.initializers import ClusterInitializer, KMeansPartitioner, MultiStartInitializer, RefiningInitializer
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.scenarios import example_sample

GRID = np.linspace(-20, 20, 161)


def example_truth() -> MixtureModel:
    """The true mixture of the example sample."""

    normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
    expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)
    return MixtureModel(
        [normal(mu=-8, sigma=1), normal(mu=-3, sigma=0.7), expon(lambda_=0.5)], weights=[2 / 7, 2 / 7, 3 / 7]
    )


def assert_close_to_truth(mixture: MixtureModel) -> None:
    """Check components 1 and 2 and the weights of a fitted mixture against the truth of the example sample."""

    assert mixture[1].parameters["mu"] == pytest.approx(-3, abs=0.2)
    assert mixture[2].parameters["lambda_"] == pytest.approx(0.5, abs=0.06)
    assert list(mixture.weights) == pytest.approx([2 / 7, 2 / 7, 3 / 7], abs=0.01)


def assert_families(mixture: MixtureModel) -> None:
    """Check that the components go in the order of ``[normal, normal, expon]``."""

    assert [component.family_name for component in mixture] == ["Normal", "Normal", "Exponential"]


@pytest.mark.filterwarnings("error")
def test_first_example_selects_a_start_from_which_em_recovers_the_truth() -> None:
    """The best of ten runs by the likelihood of the start leads the final EM to the true mixture."""

    X = example_sample()

    normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
    expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

    initializer = ClusterInitializer(KMeansPartitioner())

    # The best of 10 runs by the likelihood of the starting mixture.
    start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)

    result = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

    assert result.n_components == 3  # noqa: PLR2004
    assert_families(result)
    assert result[0].parameters["mu"] == pytest.approx(-8, abs=0.2)
    assert_close_to_truth(result)
    assert result.loglikelihood(X) >= start.loglikelihood(X)
    assert result.loglikelihood(X) >= example_truth().loglikelihood(X)


@pytest.mark.filterwarnings("error")
def test_second_example_selects_a_start_not_worse_than_the_first() -> None:
    """The best of ten short-EM runs is not worse than the first example and leads EM to the truth."""

    X = example_sample()

    normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
    expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

    initializer = ClusterInitializer(KMeansPartitioner())

    # The best of 10 runs by the likelihood of the starting mixture.
    start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)
    first_example_score = start.loglikelihood(X)

    # The best of 10 runs by the result of a short EM.
    short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
    refined = RefiningInitializer(initializer, short_em)
    start = MultiStartInitializer(refined, n_init=10).initialize(X, [normal, normal, expon], random_state=42)

    result = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

    assert start.loglikelihood(X) >= first_example_score
    assert result.n_components == 3  # noqa: PLR2004
    assert_families(result)
    assert result[0].parameters["mu"] == pytest.approx(-8, abs=0.2)
    assert_close_to_truth(result)
    assert result.loglikelihood(X) >= start.loglikelihood(X)
    assert result.loglikelihood(X) >= example_truth().loglikelihood(X)


@pytest.mark.filterwarnings("error")
def test_both_examples_lead_the_final_em_to_the_same_mixture() -> None:
    """The final EM reaches the same mixture from the starts of both examples."""

    X = example_sample()

    normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
    expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

    initializer = ClusterInitializer(KMeansPartitioner())

    # The best of 10 runs by the likelihood of the starting mixture.
    start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)
    first = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

    # The best of 10 runs by the result of a short EM.
    short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
    refined = RefiningInitializer(initializer, short_em)
    start = MultiStartInitializer(refined, n_init=10).initialize(X, [normal, normal, expon], random_state=42)
    second = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

    assert first.loglikelihood(X) == pytest.approx(second.loglikelihood(X), abs=1e-6)
    for j in range(3):
        assert first[j].parameters == pytest.approx(second[j].parameters, abs=1e-6)
    assert list(first.weights) == pytest.approx(list(second.weights), abs=1e-6)


@pytest.mark.filterwarnings("error")
def test_fixed_mean_survives_the_chain_of_the_second_example() -> None:
    """A mean fixed through a view stays exact through the chain of the second example."""

    X = example_sample()

    normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
    expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

    initializer = ClusterInitializer(KMeansPartitioner())

    # The best of 10 runs by the result of a short EM.
    short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
    refined = RefiningInitializer(initializer, short_em)
    start = MultiStartInitializer(refined, n_init=10).initialize(
        X, [normal.view(mu=-8), normal, expon], random_state=42
    )

    result = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

    assert "mu" not in result[0].parameters
    expected = normal(mu=-8, **result[0].parameters)
    density = np.asarray(result[0].calculate_characteristic(CharacteristicName.PDF, GRID))
    assert density == pytest.approx(
        np.asarray(expected.calculate_characteristic(CharacteristicName.PDF, GRID)), rel=1e-12
    )
    assert_close_to_truth(result)
