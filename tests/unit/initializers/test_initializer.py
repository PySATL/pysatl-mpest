"""Unit tests for the template method of ``Initializer``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.initializers.exceptions import InitializationError
from pysatl_mpest.initializers.initializer import Initializer
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.estimator_data import exponential, normal
from tests.helpers.families import family
from tests.mocks.distributions import ScriptedLpdfDistribution

type Build = Callable[[UnivariateFloatArray, Sequence[ParametricFamily], np.random.Generator], MixtureModel]
type Case = Callable[[], tuple[list[ParametricFamily], MixtureModel]]


def normal_family() -> ParametricFamily:
    """The registered normal family."""

    return family(FamilyName.NORMAL)


def exponential_family() -> ParametricFamily:
    """The registered exponential family."""

    return family(FamilyName.EXPONENTIAL)


def good_mixture() -> MixtureModel:
    """A valid mixture of one normal and one exponential component."""

    return MixtureModel([normal(0.0, 1.0), exponential(1.0)], [0.5, 0.5])


class ScriptedInitializer(Initializer):
    """Initializer whose result is scripted by the test."""

    def __init__(self, build: Build) -> None:
        self._build = build
        self.calls: list[tuple[UnivariateFloatArray, Sequence[ParametricFamily], np.random.Generator]] = []

    def _initialize(
        self, X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
    ) -> MixtureModel:
        self.calls.append((X, families, rng))
        return self._build(X, families, rng)


def returning(mixture: MixtureModel) -> ScriptedInitializer:
    """An initializer that always returns the given mixture."""

    def build(X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator) -> MixtureModel:
        return mixture

    return ScriptedInitializer(build)


def drawing() -> ScriptedInitializer:
    """An initializer whose component mean is drawn from the generator it receives."""

    def build(X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator) -> MixtureModel:
        return MixtureModel([normal(float(rng.normal()), 1.0)])

    return ScriptedInitializer(build)


def normal_like(lpdf_values: ArrayLike) -> ParametricFamilyDistribution:
    """A component that has the key of a normal family and scripted log-densities."""

    component = ScriptedLpdfDistribution(lpdf_values)
    component.family_name = "Normal"
    component.parametrization_name = "meanStd"
    return cast(ParametricFamilyDistribution, component)


class TestInput:
    """Checks and conversion of the sample and families before the initializer step."""

    def test_converts_sample_and_passes_same_families(self) -> None:
        families = [normal_family(), exponential_family()]
        good = good_mixture()
        initializer = returning(good)

        result = initializer.initialize([1, 2, 3], families)

        assert result is good
        assert len(initializer.calls) == 1
        received_X, received_families, _ = initializer.calls[0]
        assert received_X.dtype == np.float64
        assert received_X.ndim == 1
        np.testing.assert_array_equal(received_X, [1.0, 2.0, 3.0])
        assert received_families is families

    def test_passes_a_copy_of_the_caller_array(self) -> None:
        sample = np.array([1.0, 2.0, 3.0])
        initializer = returning(good_mixture())

        initializer.initialize(sample, [normal_family(), exponential_family()])

        received_X, _, _ = initializer.calls[0]
        assert received_X is not sample
        received_X[0] = 100.0
        np.testing.assert_array_equal(sample, [1.0, 2.0, 3.0])

    def test_rejects_empty_families(self) -> None:
        initializer = returning(good_mixture())

        with pytest.raises(ValueError, match="^families must not be empty"):
            initializer.initialize([1.0, 2.0], [])

        assert initializer.calls == []

    @pytest.mark.parametrize(
        ("make_families", "match"),
        [
            pytest.param(
                lambda: [normal(0.0, 1.0), exponential_family()],
                r"^families\[0\] must be a ParametricFamily, got ParametricFamilyDistribution",
                id="component",
            ),
            pytest.param(
                lambda: [normal_family(), FamilyName.EXPONENTIAL],
                r"^families\[1\] must be a ParametricFamily, got FamilyName",
                id="family-name",
            ),
        ],
    )
    def test_rejects_element_that_is_not_family(self, make_families: Callable[[], list[object]], match: str) -> None:
        initializer = returning(good_mixture())
        families = cast(Sequence[ParametricFamily], make_families())

        with pytest.raises(TypeError, match=match):
            initializer.initialize([1.0, 2.0], families)

        assert initializer.calls == []

    @pytest.mark.parametrize(
        ("sample", "match"),
        [
            pytest.param(
                [[1.0], [2.0], [3.0]],
                r"X must be one-dimensional, got an array of shape \(3, 1\)",
                id="column",
            ),
            pytest.param(1.0, r"X must be one-dimensional, got an array of shape \(\)", id="scalar"),
        ],
    )
    def test_rejects_sample_that_is_not_one_dimensional(self, sample: ArrayLike, match: str) -> None:
        initializer = returning(good_mixture())

        with pytest.raises(ValueError, match=match):
            initializer.initialize(sample, [normal_family(), exponential_family()])

        assert initializer.calls == []

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="positive-infinity"),
            pytest.param(-np.inf, id="negative-infinity"),
        ],
    )
    def test_rejects_non_finite_sample_values(self, bad_value: float) -> None:
        initializer = returning(good_mixture())

        with pytest.raises(ValueError, match="^X must contain only finite values"):
            initializer.initialize([1.0, bad_value, 3.0], [normal_family(), exponential_family()])

        assert initializer.calls == []

    @pytest.mark.parametrize(
        ("sample", "make_families", "match"),
        [
            pytest.param(
                [1.0],
                lambda: [normal_family(), exponential_family()],
                "X has 1 points, fewer than the 2 families",
                id="fewer-points",
            ),
            pytest.param(
                [],
                lambda: [normal_family()],
                "X has 0 points, fewer than the 1 families",
                id="empty-sample",
            ),
        ],
    )
    def test_rejects_sample_with_fewer_points_than_families(
        self, sample: ArrayLike, make_families: Callable[[], list[ParametricFamily]], match: str
    ) -> None:
        initializer = returning(good_mixture())

        with pytest.raises(ValueError, match=match):
            initializer.initialize(sample, make_families())

        assert initializer.calls == []

    def test_accepts_as_many_points_as_families(self) -> None:
        good = good_mixture()

        result = returning(good).initialize([1.0, 2.0], [normal_family(), exponential_family()])

        assert result is good


class TestRandomState:
    """Creation and passing of the random generator."""

    def test_integer_seed_reproduces_mixture_independently_of_previous_calls(self) -> None:
        initializer = drawing()
        families = [normal_family()]

        first = initializer.initialize([1.0], families, random_state=7)
        other = initializer.initialize([1.0], families, random_state=8)
        repeated = initializer.initialize([1.0], families, random_state=7)

        assert first == repeated
        assert first != other

    def test_generator_is_used_in_place_and_advances(self) -> None:
        generator = np.random.default_rng(7)
        initializer = drawing()
        families = [normal_family()]

        first = initializer.initialize([1.0], families, random_state=generator)
        second = initializer.initialize([1.0], families, random_state=generator)

        assert first != second
        assert len(initializer.calls) == 2  # noqa: PLR2004
        assert initializer.calls[0][2] is generator
        assert initializer.calls[1][2] is generator

        reference = initializer.initialize([1.0], families, random_state=7)

        assert first == reference

    def test_without_seed_every_call_uses_fresh_entropy(self) -> None:
        initializer = drawing()
        families = [normal_family()]

        first = initializer.initialize([1.0], families)
        second = initializer.initialize([1.0], families)

        assert first != second
        assert len(initializer.calls) == 2  # noqa: PLR2004
        assert isinstance(initializer.calls[0][2], np.random.Generator)
        assert isinstance(initializer.calls[1][2], np.random.Generator)


class TestResult:
    """Checks of the mixture returned by the initializer step."""

    @pytest.mark.parametrize(
        "make_case",
        [
            pytest.param(
                lambda: (
                    [normal_family(), exponential_family()],
                    MixtureModel([normal(0.0, 1.0), exponential(1.0)]),
                ),
                id="plain",
            ),
            pytest.param(
                lambda: (
                    [normal_family().view(mu=0)],
                    MixtureModel([normal_family().view(mu=0.0)(sigma=1.0)]),
                ),
                id="fixed-parameter",
            ),
            pytest.param(
                lambda: (
                    [normal_family().view(parametrization_name="meanVar")],
                    MixtureModel([normal_family().view(parametrization_name="meanVar")(mu=0.0, var=4.0)]),
                ),
                id="view-parametrization",
            ),
            pytest.param(
                lambda: (
                    [normal_family().view(parametrization_name="meanVar")],
                    MixtureModel([normal_family()(parametrization_name="meanVar", mu=0.0, var=4.0)]),
                ),
                id="plain-family-in-view-parametrization",
            ),
        ],
    )
    def test_accepts_mixture_from_given_families_in_their_order(self, make_case: Case) -> None:
        families, mixture = make_case()

        result = returning(mixture).initialize([1.0, 2.0, 3.0], families)

        assert result is mixture

    @pytest.mark.parametrize(
        ("make_case", "match"),
        [
            pytest.param(
                lambda: ([normal_family(), exponential_family()], MixtureModel([normal(0.0, 1.0)])),
                "^The initializer returned 1 components for 2 families",
                id="fewer",
            ),
            pytest.param(
                lambda: ([normal_family()], MixtureModel([normal(0.0, 1.0), normal(5.0, 1.0)])),
                "^The initializer returned 2 components for 1 families",
                id="more",
            ),
        ],
    )
    def test_rejects_other_number_of_components(self, make_case: Case, match: str) -> None:
        families, mixture = make_case()

        with pytest.raises(ValueError, match=match):
            returning(mixture).initialize([1.0, 2.0, 3.0], families)

    @pytest.mark.parametrize(
        ("make_case", "match"),
        [
            pytest.param(
                lambda: (
                    [normal_family(), exponential_family()],
                    MixtureModel([exponential(1.0), normal(0.0, 1.0)]),
                ),
                r"^Component 0 of the result is not from families\[0\]: expected family Normal in parametrization "
                r"meanStd with fixed parameters \{\}, got family Exponential in parametrization rate with fixed "
                r"parameters \{\}\.$",
                id="first-component",
            ),
            pytest.param(
                lambda: (
                    [normal_family(), exponential_family()],
                    MixtureModel([normal(0.0, 1.0), normal(5.0, 1.0)]),
                ),
                r"^Component 1 of the result is not from families\[1\]: expected family Exponential in parametrization "
                r"rate .*, got family Normal in parametrization meanStd",
                id="second-component",
            ),
        ],
    )
    def test_rejects_component_from_wrong_family(self, make_case: Case, match: str) -> None:
        families, mixture = make_case()

        with pytest.raises(ValueError, match=match):
            returning(mixture).initialize([1.0, 2.0, 3.0], families)

    @pytest.mark.parametrize(
        ("make_case", "match"),
        [
            pytest.param(
                lambda: ([normal_family().view(mu=0.0)], MixtureModel([normal(0.0, 1.0)])),
                r"expected family Normal in parametrization meanStd with fixed parameters \{'mu': 0\.0\}, got family "
                r"Normal in parametrization meanStd with fixed parameters \{\}",
                id="fixed-parameter-lost",
            ),
            pytest.param(
                lambda: (
                    [normal_family().view(mu=5.0)],
                    MixtureModel([normal_family().view(mu=0.0)(sigma=1.0)]),
                ),
                r"with fixed parameters \{'mu': 5\.0\}, got family Normal in parametrization meanStd with fixed "
                r"parameters \{'mu': 0\.0\}",
                id="other-fixed-value",
            ),
        ],
    )
    def test_rejects_component_with_wrong_fixed_parameters(self, make_case: Case, match: str) -> None:
        families, mixture = make_case()

        with pytest.raises(ValueError, match=match):
            returning(mixture).initialize([1.0, 2.0, 3.0], families)

    def test_rejects_component_in_wrong_parametrization(self) -> None:
        families = [normal_family()]
        mixture = MixtureModel([normal_family()(parametrization_name="meanVar", mu=0.0, var=1.0)])

        with pytest.raises(
            ValueError,
            match=r"expected family Normal in parametrization meanStd with fixed parameters \{\}, got family "
            r"Normal in parametrization meanVar with fixed parameters \{\}",
        ):
            returning(mixture).initialize([1.0, 2.0, 3.0], families)

    def test_uncovered_sample_point_is_initialization_error_not_value_error(self) -> None:
        families = [exponential_family(), exponential_family()]
        mixture = MixtureModel([exponential(0.5), exponential(1 / 7)])

        with pytest.raises(
            InitializationError,
            match=r"^The log-likelihood of the starting mixture on the sample is -inf\. 1 of 4 sample points are "
            r"not covered by any component, for example -1\.0\.$",
        ) as exc_info:
            returning(mixture).initialize([-1.0, 5.0, 6.0, 8.0], families)

        assert not isinstance(exc_info.value, ValueError)

    def test_message_contains_at_most_three_example_points(self) -> None:
        families = [exponential_family(), exponential_family()]
        mixture = MixtureModel([exponential(1.0), exponential(0.5)])

        with pytest.raises(
            InitializationError,
            match=r"4 of 6 sample points are not covered by any component, for example -1\.0, -3\.4, -7\.0\.$",
        ) as exc_info:
            returning(mixture).initialize([-1.0, 5.0, 2.0, -3.4, -7.0, -8.0], families)

        assert "-8.0" not in str(exc_info.value)

    def test_point_covered_by_another_component_does_not_matter(self) -> None:
        good = good_mixture()

        result = returning(good).initialize([-3.4, 0.0, 2.0], [normal_family(), exponential_family()])

        assert result is good

    @pytest.mark.parametrize(
        ("first_lpdf", "likelihood_text"),
        [
            pytest.param(np.nan, r"nan\.", id="nan"),
            pytest.param(np.inf, r"inf\.", id="positive-infinity"),
        ],
    )
    def test_undefined_density_is_initialization_error_without_uncovered_points(
        self, first_lpdf: float, likelihood_text: str
    ) -> None:
        families = [normal_family()]
        mixture = MixtureModel([normal_like([first_lpdf, -1.0, -1.0])])

        with pytest.raises(
            InitializationError,
            match=rf"^The log-likelihood of the starting mixture on the sample is {likelihood_text} The density of "
            r"the mixture is undefined at 1 of 3 sample points, for example 10\.0\.$",
        ) as exc_info:
            returning(mixture).initialize([10.0, 20.0, 30.0], families)

        assert "not covered" not in str(exc_info.value)

    def test_uncovered_and_undefined_points_are_reported_separately(self) -> None:
        families = [normal_family()]
        mixture = MixtureModel([normal_like([np.nan, -np.inf, -1.0])])

        with pytest.raises(
            InitializationError,
            match=r"^The log-likelihood of the starting mixture on the sample is nan\. 1 of 3 sample points are "
            r"not covered by any component, for example 20\.0\. The density of the mixture is undefined at 1 of 3 "
            r"sample points, for example 10\.0\.$",
        ):
            returning(mixture).initialize([10.0, 20.0, 30.0], families)

    def test_wrong_family_error_takes_precedence_over_coverage(self) -> None:
        families = [normal_family(), exponential_family()]
        mixture = MixtureModel([exponential(1.0), exponential(0.5)])

        with pytest.raises(ValueError, match=r"^Component 0 of the result is not from families\[0\]"):
            returning(mixture).initialize([-1.0, 5.0], families)

    def test_error_from_initializer_step_passes_through_as_same_object(self) -> None:
        error = InitializationError("boom")

        def build(
            X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
        ) -> MixtureModel:
            raise error

        initializer = ScriptedInitializer(build)

        with pytest.raises(InitializationError, match="^boom$") as exc_info:
            initializer.initialize([1.0], [normal_family()])

        assert exc_info.value is error

    @pytest.mark.filterwarnings("error")
    def test_infinite_likelihood_of_finite_point_densities_has_no_example_points(self) -> None:
        families = [normal_family()]
        mixture = MixtureModel([normal_like([-1e308, -1e308, -1.0])])

        with pytest.raises(
            InitializationError, match=r"^The log-likelihood of the starting mixture on the sample is -inf\.$"
        ):
            returning(mixture).initialize([10.0, 20.0, 30.0], families)
