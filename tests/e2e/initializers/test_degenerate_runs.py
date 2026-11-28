"""End-to-end scenarios: the multistart skips runs with degenerate clusters, and they never win."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.initializers import (
    ClusterInitializer,
    ComponentEstimationError,
    InitializationError,
    MultiStartInitializer,
    RandomCentersPartitioner,
)
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.scenarios import make_em


def stuck_sensor() -> UnivariateFloatArray:
    """Two wide normal modes and 40 readings of a stuck sensor: the example of the initializers design."""

    rng = np.random.default_rng(42)
    return np.concatenate(
        [rng.normal(0, 1000, 1000), rng.normal(6000, 1500, 1000), 15000 + rng.uniform(-0.001, 0.001, 40)]
    )


def outlier_sample() -> UnivariateFloatArray:
    """A sample from one normal component with a single distant outlier."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 20), [100.0]])


def failed_runs(initializer: ClusterInitializer, X: UnivariateFloatArray, seed: int) -> list[InitializationError]:
    """Errors of the failed runs of ten manual runs with the children of the seed."""

    families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL)]
    errors = []
    for child in np.random.default_rng(seed).spawn(10):
        try:
            initializer.initialize(X, families, random_state=child)
        except InitializationError as error:
            errors.append(error)
    return errors


def test_runs_with_a_too_narrow_cluster_fail_and_never_win() -> None:
    """Runs with a too narrow cluster are skipped although such a start would win by likelihood."""

    X = stuck_sensor()
    initializer = ClusterInitializer(RandomCentersPartitioner())
    errors = failed_runs(initializer, X, 13)
    assert errors
    assert all(isinstance(error, ComponentEstimationError) for error in errors)
    assert all("interquartile width" in str(error) for error in errors)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = MultiStartInitializer(initializer, 10).initialize(
            X, [family(FamilyName.NORMAL), family(FamilyName.NORMAL)], random_state=13
        )

    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning
    assert str(caught[0].message) == f"{len(errors)} of 10 runs failed; the last reason: {errors[-1]}"
    assert all(component.parameters["sigma"] > 100 for component in result)  # noqa: PLR2004

    normal = family(FamilyName.NORMAL)
    ordinary, stuck = X[:2000], X[2000:]
    narrow = MixtureModel(
        [normal(mu=ordinary.mean(), sigma=np.std(ordinary)), normal(mu=stuck.mean(), sigma=np.std(stuck))],
        weights=[2000 / 2040, 40 / 2040],
    )
    assert narrow.loglikelihood(X) > result.loglikelihood(X)

    fitted = make_em(30).fit(X, result)
    assert fitted[0].parameters["mu"] == pytest.approx(0, abs=500)
    assert fitted[1].parameters["mu"] == pytest.approx(6000, abs=500)


def test_runs_with_a_too_small_cluster_are_skipped() -> None:
    """Runs with a too small cluster are skipped with one warning, and both weights of the result are large."""

    X = outlier_sample()
    initializer = ClusterInitializer(RandomCentersPartitioner())
    errors = failed_runs(initializer, X, 0)
    assert 0 < len(errors) < 10  # noqa: PLR2004
    assert all(str(error).startswith("The partition has clusters with fewer than 4 points") for error in errors)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = MultiStartInitializer(initializer, 10).initialize(
            X, [family(FamilyName.NORMAL), family(FamilyName.NORMAL)], random_state=0
        )

    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning
    assert str(caught[0].message) == f"{len(errors)} of 10 runs failed; the last reason: {errors[-1]}"
    assert all(weight >= 4 / 21 for weight in result.weights)
