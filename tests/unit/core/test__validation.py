"""Unit tests for ``sample_coverage`` with scripted component log-densities."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import CharacteristicName
from pysatl_mpest.core._validation import sample_coverage
from pysatl_mpest.core.mixture import MixtureModel

from tests.mocks.distributions import MockContinuousDistribution


class ScriptedLpdfDistribution(MockContinuousDistribution):
    """Mock component that returns pre-set log-densities, one value per sample point."""

    def __init__(self, lpdf_values: ArrayLike) -> None:
        super().__init__()
        self._lpdf_values = np.asarray(lpdf_values, dtype=np.float64)

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        if name == CharacteristicName.LPDF:
            return self._lpdf_values
        raise NotImplementedError


def mixture_of(*lpdf_columns: ArrayLike, log_weights: ArrayLike | None = None) -> MixtureModel:
    """A mixture whose components return the given log-densities, one column per component."""

    components = [cast(ParametricFamilyDistribution, ScriptedLpdfDistribution(column)) for column in lpdf_columns]
    mixture = MixtureModel(components)

    return mixture if log_weights is None else mixture.with_log_weights(log_weights)


def test_reports_nothing_when_log_density_is_finite_everywhere() -> None:
    mixture = mixture_of([-1.0, -2.0, -3.0], [-1.5, -0.5, -4.0])

    coverage = sample_coverage(mixture, np.array([10.0, 20.0, 30.0]))

    np.testing.assert_array_equal(coverage.uncovered, [])
    np.testing.assert_array_equal(coverage.undefined, [])


def test_reports_point_uncovered_only_when_no_component_covers_it() -> None:
    mixture = mixture_of(
        [-np.inf, -np.inf, -1.0, -np.inf],
        [-np.inf, -2.0, -np.inf, -np.inf],
    )

    coverage = sample_coverage(mixture, np.array([10.0, 20.0, 30.0, 40.0]))

    np.testing.assert_array_equal(coverage.uncovered, [0, 3])
    np.testing.assert_array_equal(coverage.undefined, [])


def test_reports_nan_point_as_undefined_not_uncovered() -> None:
    mixture = mixture_of(
        [np.nan, -1.0, -np.inf, np.nan],
        [-1.0, -1.0, -np.inf, -np.inf],
    )

    coverage = sample_coverage(mixture, np.array([10.0, 20.0, 30.0, 40.0]))

    np.testing.assert_array_equal(coverage.undefined, [0, 3])
    np.testing.assert_array_equal(coverage.uncovered, [2])


def test_returns_one_dimensional_integer_indices() -> None:
    mixture = mixture_of(
        [-np.inf, -np.inf, -1.0, -np.inf],
        [-np.inf, -2.0, -np.inf, -np.inf],
    )

    coverage = sample_coverage(mixture, np.array([10.0, 20.0, 30.0, 40.0]))

    for indices in (coverage.uncovered, coverage.undefined):
        assert indices.ndim == 1
        assert indices.dtype.kind == "i"


def test_reports_nothing_for_empty_sample() -> None:
    mixture = mixture_of([], [])

    coverage = sample_coverage(mixture, np.array([]))

    assert coverage.uncovered.shape == (0,)
    assert coverage.undefined.shape == (0,)


def test_component_with_zero_weight_covers_nothing() -> None:
    mixture = mixture_of([-np.inf, -1.0], [-2.0, -np.inf], log_weights=[0.0, -np.inf])

    coverage = sample_coverage(mixture, np.array([10.0, 20.0]))

    np.testing.assert_array_equal(coverage.uncovered, [0])
    np.testing.assert_array_equal(coverage.undefined, [])


def test_reports_pos_inf_point_as_undefined() -> None:
    mixture = mixture_of([np.inf, -1.0], [-1.0, -1.0])

    coverage = sample_coverage(mixture, np.array([10.0, 20.0]))

    np.testing.assert_array_equal(coverage.undefined, [0])
    np.testing.assert_array_equal(coverage.uncovered, [])
