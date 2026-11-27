"""Unit tests for the lazy component table."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel, _components
from pysatl_mpest.initializers.cluster._table import _lazy
from pysatl_mpest.initializers.cluster._table._lazy import LazyComponentTable
from pysatl_mpest.initializers.cluster._table._pair import PairEvaluation
from pysatl_mpest.initializers.cluster.component_estimators.base import ComponentEstimator
from pysatl_mpest.initializers.exceptions import ComponentEstimationError
from pysatl_mpest.typings import MultivariateFloatArray, UnivariateFloatArray

from tests.helpers.families import family

R = "scripted reason"


def scripted_component(mu: float = 0.0) -> ParametricFamilyDistribution:
    """A new ready component; the tests use it only as an object with identity."""

    return family(FamilyName.NORMAL).distribution(mu=mu, sigma=1.0)


class ScriptedPairEvaluation:
    """Double of ``evaluate_pair``: returns scripted results by family key and column and records the calls."""

    def __init__(
        self,
        H: MultivariateFloatArray,
        results: dict[tuple[_components.FamilyKey, int], PairEvaluation] | None = None,
    ) -> None:
        self.H = H
        self.results = dict(results or {})
        self.calls: list[tuple[Any, Any, Any, Any, Any]] = []

    def __call__(
        self,
        X: UnivariateFloatArray,
        h: UnivariateFloatArray,
        family: ParametricFamily,
        component_estimator: ComponentEstimator,
        scale: float,
    ) -> PairEvaluation:
        self.calls.append((X, h, family, component_estimator, scale))

        column = next(c for c in range(self.H.shape[1]) if np.array_equal(self.H[:, c], h))
        key = (_components.family_key(family), column)
        if key in self.results:
            return self.results[key]
        return PairEvaluation(scripted_component(), -1.0, None)


class UnusedEstimator(ComponentEstimator):
    """Component estimator double that must never be called: the table only passes it on."""

    def supports(self, family: ParametricFamily) -> bool:
        raise AssertionError("The table must not call the component estimator.")

    def estimate(self, X: UnivariateFloatArray, h: UnivariateFloatArray, family: ParametricFamily) -> dict[str, float]:
        raise AssertionError("The table must not call the component estimator.")


@pytest.fixture
def normal() -> ParametricFamily:
    """The Normal family without a view."""

    return family(FamilyName.NORMAL)


@pytest.fixture
def expon() -> ParametricFamily:
    """The Exponential family without a view."""

    return family(FamilyName.EXPONENTIAL)


def univariate(values: list[float]) -> UnivariateFloatArray:
    """One-dimensional ``float64`` array from literal values."""

    return np.asarray(values, dtype=np.float64)


def standard_partition() -> tuple[UnivariateFloatArray, MultivariateFloatArray]:
    """The standard read-only sample and partition of this file: clusters of sizes 3, 1 and 3."""

    X = univariate([1.0, 2.0, 10.0, 20.0, 21.0, 22.0, 3.0])
    H = np.eye(3)[[0, 0, 1, 2, 2, 2, 0]]
    X.flags.writeable = False
    H.flags.writeable = False
    return X, H


def point_partition(k: int) -> tuple[UnivariateFloatArray, MultivariateFloatArray]:
    """A read-only partition of the points ``0, ..., k - 1`` into one-point clusters."""

    X = np.arange(k, dtype=np.float64)
    H = np.eye(k, dtype=np.float64)
    X.flags.writeable = False
    H.flags.writeable = False
    return X, H


def make_table(
    monkeypatch: pytest.MonkeyPatch,
    X: UnivariateFloatArray,
    H: MultivariateFloatArray,
    families: Sequence[ParametricFamily],
    results: dict[tuple[_components.FamilyKey, int], PairEvaluation] | None = None,
    estimator: ComponentEstimator | None = None,
) -> tuple[LazyComponentTable, ScriptedPairEvaluation]:
    """A table whose pair evaluation is the scripted double."""

    pairs = ScriptedPairEvaluation(H, results)
    monkeypatch.setattr(_lazy, "evaluate_pair", pairs)
    if estimator is None:
        estimator = UnusedEstimator()
    return LazyComponentTable(X, H, families, estimator), pairs


class TestStructure:
    """Storage of the input, groups of identical families and cluster means."""

    def test_stores_the_input_objects_and_counts_families(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        families = [normal, expon, normal]

        table, _ = make_table(monkeypatch, X, H, families)

        assert table.k == 3  # noqa: PLR2004
        assert table.X is X
        assert table.H is H
        assert type(table.families) is tuple
        assert len(table.families) == 3  # noqa: PLR2004
        assert all(table.families[j] is families[j] for j in range(3))

    @pytest.mark.parametrize(
        ("make_families", "expected"),
        [
            pytest.param(
                lambda: [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)],
                ((0, 2), (1,)),
                id="repeated-family",
            ),
            pytest.param(
                lambda: [
                    family(FamilyName.NORMAL).view(mu=0),
                    family(FamilyName.NORMAL).view(mu=5),
                    family(FamilyName.NORMAL),
                ],
                ((0,), (1,), (2,)),
                id="different-fixed-values",
            ),
            pytest.param(
                lambda: [family(FamilyName.NORMAL).view(mu=0), family(FamilyName.NORMAL).view(mu=0.0)],
                ((0, 1),),
                id="equal-fixed-values",
            ),
            pytest.param(
                lambda: [family(FamilyName.NORMAL), family(FamilyName.NORMAL).view()],
                ((0, 1),),
                id="empty-view",
            ),
        ],
    )
    def test_groups_indices_of_identical_families_in_order_of_first_occurrence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_families: Callable[[], list[ParametricFamily]],
        expected: tuple[tuple[int, ...], ...],
    ) -> None:
        families = make_families()
        X, H = point_partition(len(families))

        table, _ = make_table(monkeypatch, X, H, families)

        assert table.groups() == expected

    def test_cluster_means_are_read_only_float64_means_of_cluster_points(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        table, _ = make_table(monkeypatch, X, H, [normal, expon, normal])

        means = table.cluster_means()

        assert np.array_equal(means, [2.0, 10.0, 21.0])
        assert means.dtype == np.float64
        assert means.shape == (3,)
        assert not means.flags.writeable

    def test_creation_structure_and_means_do_not_evaluate_pairs(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()

        table, pairs = make_table(monkeypatch, X, H, [normal, expon, normal])
        assert table.k == 3  # noqa: PLR2004
        table.groups()
        table.cluster_means()

        assert pairs.calls == []


class TestPairs:
    """Lazy evaluation, normalization and reading of the records of pairs."""

    def test_first_request_passes_sample_column_family_estimator_and_scale(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        families = [normal, expon, normal]
        estimator = UnusedEstimator()

        table, pairs = make_table(monkeypatch, X, H, families, estimator=estimator)
        table.loglikelihood(1, 2)

        assert len(pairs.calls) == 1
        called_X, called_h, called_family, called_estimator, scale = pairs.calls[0]
        assert called_X is X
        assert np.array_equal(called_h, H[:, 2])
        assert called_family is families[1]
        assert called_estimator is estimator
        assert scale == float(np.std(X))
        assert type(scale) is float

    @pytest.mark.parametrize(
        "first",
        [
            pytest.param("loglikelihood", id="loglikelihood-first"),
            pytest.param("failure", id="failure-first"),
            pytest.param("component", id="component-first"),
        ],
    )
    def test_is_evaluated_once_whichever_method_requests_it(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily, first: str
    ) -> None:
        X, H = standard_partition()
        table, pairs = make_table(monkeypatch, X, H, [normal, expon, normal])

        getattr(table, first)(1, 2)
        for _ in range(2):
            table.loglikelihood(1, 2)
            table.failure(1, 2)
            table.component(1, 2)

        assert len(pairs.calls) == 1

    def test_identical_families_share_one_entry(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(normal), 1): PairEvaluation(scripted_component(), -4.25, None)}
        table, pairs = make_table(monkeypatch, X, H, [normal, expon, normal], results)

        assert table.loglikelihood(0, 1) == -4.25  # noqa: PLR2004
        assert table.loglikelihood(2, 1) == -4.25  # noqa: PLR2004

        assert len(pairs.calls) == 1

    def test_entries_are_shared_by_family_key_and_cluster(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        table, pairs = make_table(monkeypatch, X, H, [normal, expon, normal])

        for j in range(3):
            for c in range(3):
                table.loglikelihood(j, c)

        assert len(pairs.calls) == 6  # noqa: PLR2004

    def test_families_with_different_fixed_parameters_do_not_share_entries(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily
    ) -> None:
        families = [normal.view(mu=0), normal.view(mu=5)]
        X, H = point_partition(2)
        results = {
            (_components.family_key(families[0]), 0): PairEvaluation(scripted_component(), -1.0, None),
            (_components.family_key(families[1]), 0): PairEvaluation(scripted_component(), -2.0, None),
        }
        table, pairs = make_table(monkeypatch, X, H, families, results)

        assert table.loglikelihood(0, 0) == -1.0
        assert table.loglikelihood(1, 0) == -2.0  # noqa: PLR2004

        assert len(pairs.calls) == 2  # noqa: PLR2004
        assert pairs.calls[0][2] is families[0]
        assert pairs.calls[1][2] is families[1]

    @pytest.mark.parametrize(
        ("value", "reason", "expected"),
        [
            pytest.param(-3.5, None, -3.5, id="finite"),
            pytest.param(np.nan, R, -np.inf, id="nan"),
            pytest.param(np.inf, R, -np.inf, id="positive-infinity"),
            pytest.param(-np.inf, R, -np.inf, id="negative-infinity"),
        ],
    )
    def test_finite_loglikelihood_is_returned_as_is_and_non_finite_as_minus_infinity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        normal: ParametricFamily,
        expon: ParametricFamily,
        value: float,
        reason: str | None,
        expected: float,
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(expon), 2): PairEvaluation(scripted_component(), value, reason)}
        table, _ = make_table(monkeypatch, X, H, [normal, expon, normal], results)

        assert table.loglikelihood(1, 2) == expected

    def test_failure_is_none_for_a_finite_loglikelihood(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(expon), 1): PairEvaluation(scripted_component(), -3.5, None)}
        table, _ = make_table(monkeypatch, X, H, [expon, normal, expon], results)

        assert table.failure(2, 1) is None

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="positive-infinity"),
            pytest.param(-np.inf, id="negative-infinity"),
        ],
    )
    def test_failure_of_non_finite_loglikelihood_has_numbers_in_prefix(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily, value: float
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(expon), 1): PairEvaluation(scripted_component(), value, R)}
        table, _ = make_table(monkeypatch, X, H, [expon, normal, expon], results)

        assert table.failure(2, 1) == "family 2 (Exponential), cluster 1: scripted reason"

    def test_failure_uses_the_number_of_the_requesting_family_in_a_shared_entry(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(expon), 1): PairEvaluation(None, -np.inf, R)}
        table, pairs = make_table(monkeypatch, X, H, [expon, normal, expon], results)

        assert table.failure(0, 1) == "family 0 (Exponential), cluster 1: scripted reason"
        assert table.failure(2, 1) == "family 2 (Exponential), cluster 1: scripted reason"
        assert len(pairs.calls) == 1

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            pytest.param(-3.5, None, id="estimated"),
            pytest.param(-np.inf, R, id="does-not-cover"),
        ],
    )
    def test_component_is_the_same_object_at_every_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        normal: ParametricFamily,
        expon: ParametricFamily,
        value: float,
        reason: str | None,
    ) -> None:
        X, H = standard_partition()
        component = scripted_component()
        results = {(_components.family_key(expon), 2): PairEvaluation(component, value, reason)}
        table, _ = make_table(monkeypatch, X, H, [normal, expon, normal], results)

        assert table.component(1, 2) is component
        assert table.component(1, 2) is component

    def test_component_of_not_estimated_pair_raises_a_new_error_with_failure_text(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(expon), 1): PairEvaluation(None, -np.inf, R)}
        table, pairs = make_table(monkeypatch, X, H, [expon, normal, expon], results)
        message = "family 2 (Exponential), cluster 1: scripted reason"

        with pytest.raises(ComponentEstimationError, match=rf"^{re.escape(message)}$") as first:
            table.component(2, 1)
        with pytest.raises(ComponentEstimationError, match=rf"^{re.escape(message)}$") as second:
            table.component(2, 1)

        assert type(first.value) is ComponentEstimationError
        assert first.value is not second.value
        assert str(first.value) == table.failure(2, 1)
        assert len(pairs.calls) == 1


class TestMixture:
    """Assembly of a mixture from the cached components."""

    def test_has_components_in_family_order_and_weights_in_assignment_order(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        a, b, c = scripted_component(1.0), scripted_component(2.0), scripted_component(3.0)
        results = {
            (_components.family_key(normal), 2): PairEvaluation(a, -1.0, None),
            (_components.family_key(expon), 0): PairEvaluation(b, -1.0, None),
            (_components.family_key(normal), 1): PairEvaluation(c, -1.0, None),
        }
        table, _ = make_table(monkeypatch, X, H, [normal, expon, normal], results)

        mixture = table.mixture(np.array([2, 0, 1]))

        assert isinstance(mixture, MixtureModel)
        assert len(mixture.components) == 3  # noqa: PLR2004
        assert mixture.components[0] is a
        assert mixture.components[1] is b
        assert mixture.components[2] is c
        np.testing.assert_allclose(mixture.weights, [3 / 7, 3 / 7, 1 / 7], rtol=1e-12)

    def test_does_not_evaluate_pairs_that_are_already_cached(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        table, pairs = make_table(monkeypatch, X, H, [normal, expon, normal])
        for j in range(3):
            for c in range(3):
                table.loglikelihood(j, c)
        assert len(pairs.calls) == 6  # noqa: PLR2004

        table.mixture(np.array([2, 0, 1]))

        assert len(pairs.calls) == 6  # noqa: PLR2004

    def test_evaluates_only_the_missing_selected_pairs(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        table, pairs = make_table(monkeypatch, X, H, [normal, expon, normal])

        table.mixture(np.array([2, 0, 1]))

        assert len(pairs.calls) == 3  # noqa: PLR2004

    def test_selected_not_estimated_pair_interrupts_assembly_with_its_error(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        results = {(_components.family_key(expon), 0): PairEvaluation(None, -np.inf, R)}
        table, _ = make_table(monkeypatch, X, H, [normal, expon, normal], results)
        message = "family 1 (Exponential), cluster 0: scripted reason"

        with pytest.raises(ComponentEstimationError, match=rf"^{re.escape(message)}$"):
            table.mixture(np.array([2, 0, 1]))

    def test_pair_that_does_not_cover_cluster_does_not_interrupt_assembly(
        self, monkeypatch: pytest.MonkeyPatch, normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        X, H = standard_partition()
        b = scripted_component(2.0)
        results = {(_components.family_key(expon), 0): PairEvaluation(b, -np.inf, R)}
        table, _ = make_table(monkeypatch, X, H, [normal, expon, normal], results)

        mixture = table.mixture(np.array([2, 0, 1]))

        assert isinstance(mixture, MixtureModel)
        assert mixture.components[1] is b
