"""Integration tests of the lazy component table with real pair evaluation."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel, _components
from pysatl_mpest.initializers.cluster._table._lazy import LazyComponentTable
from pysatl_mpest.initializers.cluster.component_estimators.mle_component_estimator import MLEComponentEstimator
from pysatl_mpest.initializers.exceptions import ComponentEstimationError
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.partitions import MAX_CLUSTERS, partition_inputs, random_partition

CELLS = [FamilyName.NORMAL, FamilyName.EXPONENTIAL, FamilyName.CONTINUOUS_UNIFORM]


@pytest.fixture
def normal() -> ParametricFamily:
    """The Normal family without a view."""

    return family(FamilyName.NORMAL)


@pytest.fixture
def expon() -> ParametricFamily:
    """The Exponential family without a view."""

    return family(FamilyName.EXPONENTIAL)


@pytest.fixture
def uniform() -> ParametricFamily:
    """The continuous uniform family without a view."""

    return family(FamilyName.CONTINUOUS_UNIFORM)


def make_table(
    values: Sequence[float] | UnivariateFloatArray,
    labels: Sequence[int],
    families: Sequence[ParametricFamily],
) -> LazyComponentTable:
    """A table over a hard partition given by cluster labels, with the real estimator."""

    X = np.asarray(values, dtype=np.float64)
    H = np.eye(len(families), dtype=np.float64)[np.asarray(labels)]
    X.flags.writeable = False
    H.flags.writeable = False
    return LazyComponentTable(X, H, families, MLEComponentEstimator())


def design_example_table(normal: ParametricFamily, expon: ParametricFamily) -> LazyComponentTable:
    """The table of the design example: clusters ``{4, 5, 6}`` and ``{-1, 5}``."""

    return make_table([4.0, 5.0, 6.0, -1.0, 5.0], [0, 0, 0, 1, 1], [normal.view(mu=0), expon])


def random_table(inputs: tuple[UnivariateFloatArray, int, int], names: list[FamilyName]) -> LazyComponentTable:
    """A table over a random partition with the first ``k`` of the drawn families and the real estimator."""

    X, k, seed = inputs
    H = random_partition(len(X), k, seed)
    return LazyComponentTable(X, H, [family(name) for name in names[:k]], MLEComponentEstimator())


class TestPairs:
    """States of pairs evaluated by the real estimator."""

    @pytest.mark.parametrize(
        ("j", "expected"),
        [
            pytest.param(0, {"mu": (-10.0, 0.3), "sigma": (1.0, 0.3)}, id="normal"),
            pytest.param(1, {"lambda_": (0.5, 0.05)}, id="exponential"),
            pytest.param(2, {"lower_bound": (20.0, 0.3), "upper_bound": (30.0, 0.3)}, id="uniform"),
        ],
    )
    def test_own_cluster_is_estimated_with_parameters_close_to_true_values(
        self,
        normal: ParametricFamily,
        expon: ParametricFamily,
        uniform: ParametricFamily,
        j: int,
        expected: dict[str, tuple[float, float]],
    ) -> None:
        rng = np.random.default_rng(42)
        X = np.concatenate([rng.normal(-10.0, 1.0, 100), rng.exponential(2.0, 100), rng.uniform(20.0, 30.0, 100)])
        labels = np.repeat([0, 1, 2], 100)
        table = make_table(X, labels, [normal, expon, uniform])

        assert np.isfinite(table.loglikelihood(j, j))
        assert table.failure(j, j) is None

        free = _components.free_parameters(table.component(j, j))
        for name, (value, tolerance) in expected.items():
            assert free[name] == pytest.approx(value, abs=tolerance)

    def test_not_estimated_pair_has_minus_infinity_prefixed_failure_and_component_error(
        self, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        table = make_table([-5.0, 1.0, 4.0, 6.0], [0, 0, 1, 1], [normal, expon])

        assert table.loglikelihood(1, 0) == -np.inf

        failure = table.failure(1, 0)
        assert failure is not None
        assert failure.startswith("family 1 (Exponential), cluster 0: ")
        with pytest.raises(ComponentEstimationError, match=rf"^{re.escape(failure)}$"):
            table.component(1, 0)

    def test_design_example_loglikelihoods(self, normal: ParametricFamily, expon: ParametricFamily) -> None:
        table = design_example_table(normal, expon)

        assert table.loglikelihood(0, 0) == pytest.approx(-9.124605299392378, rel=1e-12)
        assert table.loglikelihood(0, 1) == pytest.approx(-5.4028264238708825, rel=1e-12)
        assert table.loglikelihood(1, 0) == pytest.approx(-7.8283137373023015, rel=1e-12)
        assert table.loglikelihood(1, 1) == -np.inf
        assert table.failure(0, 0) is None
        assert table.failure(0, 1) is None
        assert table.failure(1, 0) is None

    def test_design_example_pair_that_does_not_cover_cluster_keeps_component_and_failure(
        self, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        table = design_example_table(normal, expon)

        assert _components.free_parameters(table.component(1, 1)) == {"lambda_": pytest.approx(0.5)}
        failure = table.failure(1, 1)
        assert failure is not None
        assert failure.startswith("family 1 (Exponential), cluster 1: ")


@settings(deadline=None)
@given(
    inputs=partition_inputs(),
    names=st.lists(st.sampled_from(CELLS), min_size=MAX_CLUSTERS, max_size=MAX_CLUSTERS),
)
def test_every_pair_has_a_consistent_state_on_random_partitions(
    inputs: tuple[UnivariateFloatArray, int, int], names: list[FamilyName]
) -> None:
    table = random_table(inputs, names)

    for j, tested_family in enumerate(table.families):
        for c in range(table.k):
            value = table.loglikelihood(j, c)

            assert np.isfinite(value) or value == -np.inf
            assert (table.failure(j, c) is None) == bool(np.isfinite(value))

            try:
                component = table.component(j, c)
            except ComponentEstimationError as error:
                assert str(error) == table.failure(j, c)
            else:
                assert _components.component_family_key(component) == _components.family_key(tested_family)


class TestIdenticalFamilies:
    """Groups and shared records of families with equal keys."""

    def test_form_one_group(self, normal: ParametricFamily, expon: ParametricFamily) -> None:
        table = make_table(
            [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [normal, normal, expon],
        )

        assert table.groups() == ((0, 1), (2,))

    def test_get_the_same_component_on_every_cluster(self, normal: ParametricFamily, expon: ParametricFamily) -> None:
        table = make_table(
            [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [normal, normal, expon],
        )

        for c in range(3):
            assert table.component(0, c) is table.component(1, c)
            assert table.component(0, c) is table.component(0, c)
            assert table.loglikelihood(0, c) == table.loglikelihood(1, c)

    def test_family_with_fixed_parameter_does_not_share_entry_and_keeps_it(self, normal: ParametricFamily) -> None:
        families = [normal.view(mu=0), normal]
        table = make_table([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], [0, 0, 0, 1, 1, 1], families)

        assert table.groups() == ((0,), (1,))
        assert _components.all_parameters(table.component(0, 0)) == {
            "mu": 0,
            "sigma": pytest.approx(2.160246899469287, rel=1e-12),
        }
        assert _components.component_family_key(table.component(0, 0)) == _components.family_key(families[0])
        assert table.component(0, 0) is not table.component(1, 0)
        assert _components.free_parameters(table.component(1, 0))["mu"] == pytest.approx(2.0)


@settings(deadline=None)
@given(
    inputs=partition_inputs(),
    names=st.lists(st.sampled_from(CELLS), min_size=MAX_CLUSTERS, max_size=MAX_CLUSTERS),
)
def test_identical_families_have_equal_loglikelihoods_on_random_partitions(
    inputs: tuple[UnivariateFloatArray, int, int], names: list[FamilyName]
) -> None:
    table = random_table(inputs, names)

    for group in table.groups():
        for c in range(table.k):
            for j in group:
                assert table.loglikelihood(j, c) == table.loglikelihood(group[0], c)


class TestMixture:
    """Assembly of a mixture by a permutation of clusters."""

    def test_components_are_estimated_on_assigned_clusters_and_belong_to_their_families(
        self, normal: ParametricFamily
    ) -> None:
        families = [normal, normal]
        table = make_table([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], [0, 0, 0, 1, 1, 1], families)

        mixture = table.mixture(np.array([1, 0]))

        assert _components.free_parameters(mixture.components[0]) == {
            "mu": pytest.approx(11.0),
            "sigma": pytest.approx(0.816496580927726, rel=1e-12),
        }
        free_second = _components.free_parameters(mixture.components[1])
        assert free_second["mu"] == pytest.approx(2.0)
        assert free_second["sigma"] == pytest.approx(0.816496580927726, rel=1e-12)
        np.testing.assert_allclose(mixture.weights, [0.5, 0.5], rtol=1e-12)
        assert mixture == MixtureModel(
            [
                normal.distribution(mu=11.0, sigma=0.816496580927726),
                normal.distribution(mu=2.0, sigma=0.816496580927726),
            ],
            [0.5, 0.5],
        )
        for j in range(2):
            assert _components.component_family_key(mixture.components[j]) == _components.family_key(families[j])

    def test_design_example_has_cluster_fractions_as_weights_in_assignment_order(
        self, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X = np.asarray([4.0, 5.0, 6.0, -1.0, 5.0], dtype=np.float64)
        table = design_example_table(normal, expon)

        mixture = table.mixture(np.array([1, 0]))

        np.testing.assert_allclose(mixture.weights, [2 / 5, 3 / 5], rtol=1e-12)
        assert mixture.loglikelihood(X) == pytest.approx(-14.353691777996962, rel=1e-12)

    def test_design_example_pair_that_does_not_cover_cluster_does_not_interrupt_assembly(
        self, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        table = design_example_table(normal, expon)

        mixture = table.mixture(np.array([0, 1]))

        assert isinstance(mixture, MixtureModel)
        assert mixture.components[1] is table.component(1, 1)
        np.testing.assert_allclose(mixture.weights, [3 / 5, 2 / 5], rtol=1e-12)

    def test_assignment_with_not_estimated_pair_interrupts_assembly(
        self, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        table = make_table([-5.0, 1.0, 4.0, 6.0], [0, 0, 1, 1], [normal, expon])

        with pytest.raises(ComponentEstimationError, match=r"^family 1 \(Exponential\), cluster 0: "):
            table.mixture(np.array([1, 0]))
