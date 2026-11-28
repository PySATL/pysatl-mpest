"""Unit tests for ``MultiStartInitializer`` with a scripted nested initializer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.initializers.cluster.partitioners.kmeans_partitioner import KMeansPartitioner
from pysatl_mpest.initializers.exceptions import ComponentEstimationError, InitializationError
from pysatl_mpest.initializers.multi_start import MultiStartInitializer
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.estimator_data import normal as normal_component
from tests.helpers.families import family
from tests.mocks.initializers import ScriptedInitializer

type Outcome = MixtureModel | Exception
type OutcomesAndBest = tuple[list[Outcome], int]


def normal_family() -> ParametricFamily:
    """The registered normal family."""

    return family(FamilyName.NORMAL)


def uniform_family() -> ParametricFamily:
    """The registered continuous uniform family."""

    return family(FamilyName.CONTINUOUS_UNIFORM)


def normal_mixture(mu: float) -> MixtureModel:
    """A one-component normal mixture with unit standard deviation."""

    return MixtureModel([normal_component(mu, 1.0)])


def uniform_mixture(component_family: ParametricFamily, lower_bound: float, upper_bound: float) -> MixtureModel:
    """A one-component continuous uniform mixture."""

    component = component_family.distribution(lower_bound=lower_bound, upper_bound=upper_bound)
    return MixtureModel([component])


def scripted(outcomes: Sequence[Outcome]) -> ScriptedInitializer:
    """An initializer that returns or raises the outcomes in the order of its calls."""

    call_number = 0

    def build(X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator) -> MixtureModel:
        nonlocal call_number
        outcome = outcomes[call_number]
        call_number += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return ScriptedInitializer(build)


def fullmatch(text: str) -> str:
    """Regex that matches the whole text literally."""

    return f"^{re.escape(text)}$"


def same_state(first: np.random.Generator, second: np.random.Generator) -> bool:
    """Whether two generators have the same bit-generator state."""

    return bool(first.bit_generator.state == second.bit_generator.state)


def one_failure_first() -> OutcomesAndBest:
    """Scenario with one unsuccessful first run."""

    return [InitializationError("a"), normal_mixture(0.0), normal_mixture(1.0)], 1


def one_failure_last() -> OutcomesAndBest:
    """Scenario with one unsuccessful last run."""

    return [normal_mixture(0.0), normal_mixture(1.0), InitializationError("a")], 0


class TestConstruction:
    """Storage and validation of the nested initializer and the number of runs."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(3, id="python-int"),
            pytest.param(np.int64(3), id="numpy-int"),
        ],
    )
    def test_stores_initializer_and_positive_integer_count_as_int(self, value: object) -> None:
        nested = scripted([normal_mixture(0.0)])

        initializer = MultiStartInitializer(nested, value)

        assert initializer.initializer is nested
        assert initializer.n_init == 3  # noqa: PLR2004
        assert type(initializer.n_init) is int

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
            pytest.param(2.5, id="float"),
            pytest.param("3", id="string"),
            pytest.param(True, id="boolean"),
            pytest.param(None, id="none"),
        ],
    )
    def test_rejects_invalid_count(self, value: object) -> None:
        nested = scripted([normal_mixture(0.0)])
        expected = f"n_init must be a positive integer, got {value!r}."

        with pytest.raises(ValueError, match=fullmatch(expected)):
            MultiStartInitializer(nested, value)

    @pytest.mark.parametrize(
        ("make_bad_initializer", "type_name"),
        [
            pytest.param(lambda: ScriptedInitializer, "ABCMeta", id="initializer-class"),
            pytest.param(lambda: KMeansPartitioner(), "KMeansPartitioner", id="partitioner"),
            pytest.param(lambda: "initializer", "str", id="string"),
        ],
    )
    def test_rejects_initializer_of_wrong_type(
        self, make_bad_initializer: Callable[[], object], type_name: str
    ) -> None:
        expected = f"initializer must be an Initializer, got {type_name}."

        with pytest.raises(TypeError, match=fullmatch(expected)):
            MultiStartInitializer(make_bad_initializer(), 3)


class TestSelection:
    """Choice of the best successful run and use of the nested initializer."""

    @pytest.mark.parametrize(
        ("mus", "best_index"),
        [
            pytest.param([0.0, 3.0, 0.5], 0, id="first"),
            pytest.param([3.0, 0.0, 0.5], 1, id="middle"),
            pytest.param([3.0, 0.5, 0.0], 2, id="last"),
        ],
    )
    def test_returns_same_object_with_largest_loglikelihood(self, mus: list[float], best_index: int) -> None:
        outcomes = [normal_mixture(mu) for mu in mus]
        initializer = MultiStartInitializer(scripted(outcomes), len(outcomes))

        result = initializer.initialize([0.0], [normal_family()], random_state=7)

        assert result is outcomes[best_index]

    @pytest.mark.parametrize(
        ("mus", "best_index"),
        [
            pytest.param([-1.0, 1.0], 0, id="two-equal"),
            pytest.param([3.0, -1.0, 1.0], 1, id="equal-after-worse"),
        ],
    )
    def test_equal_scores_are_won_by_earlier_run(self, mus: list[float], best_index: int) -> None:
        outcomes = [normal_mixture(mu) for mu in mus]
        initializer = MultiStartInitializer(scripted(outcomes), len(outcomes))

        result = initializer.initialize([0.0], [normal_family()], random_state=7)

        assert result is outcomes[best_index]

    @pytest.mark.parametrize(
        "n_init",
        [
            pytest.param(3, id="three-runs"),
            pytest.param(1, id="one-run"),
        ],
    )
    def test_runs_nested_initializer_with_checked_sample_and_same_families(self, n_init: int) -> None:
        nested = scripted([normal_mixture(0.0) for _ in range(n_init)])
        initializer = MultiStartInitializer(nested, n_init)
        families = [normal_family()]

        initializer.initialize([0.0], families, random_state=7)

        assert len(nested.calls) == n_init
        assert len({id(received_X) for received_X, _, _ in nested.calls}) == n_init
        for received_X, received_families, _ in nested.calls:
            assert received_X.dtype == np.float64
            assert not received_X.flags.writeable
            np.testing.assert_array_equal(received_X, [0.0])
            assert received_families is families

    @pytest.mark.filterwarnings("error")
    def test_successful_runs_do_not_warn(self) -> None:
        initializer = MultiStartInitializer(scripted([normal_mixture(0.0) for _ in range(3)]), 3)

        initializer.initialize([0.0], [normal_family()], random_state=7)

    def test_has_no_state_after_call(self) -> None:
        n_init = 3
        nested = scripted([normal_mixture(0.0) for _ in range(2 * n_init)])
        initializer = MultiStartInitializer(nested, n_init)
        families = [normal_family()]

        initializer.initialize([0.0], families, random_state=7)

        assert initializer.initializer is nested
        assert initializer.n_init == n_init
        assert len(nested.calls) == n_init

        initializer.initialize([0.0], families, random_state=7)

        assert len(nested.calls) == 2 * n_init


class TestGenerators:
    """Creation and reproducibility of the independent run generators."""

    @pytest.mark.parametrize(
        "make_outcomes",
        [
            pytest.param(lambda: [normal_mixture(0.0) for _ in range(3)], id="all-successful"),
            pytest.param(
                lambda: [InitializationError("a"), normal_mixture(0.0), normal_mixture(1.0)],
                id="first-unsuccessful",
            ),
        ],
    )
    def test_runs_receive_children_in_order(self, make_outcomes: Callable[[], list[Outcome]]) -> None:
        n_init = 3
        nested = scripted(make_outcomes())
        initializer = MultiStartInitializer(nested, n_init)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            initializer.initialize([0.0], [normal_family()], random_state=7)

        children = [rng for _, _, rng in nested.calls]
        reference = np.random.default_rng(7).spawn(n_init)
        base_state = np.random.default_rng(7).bit_generator.state
        assert len({id(child) for child in children}) == n_init
        for child, reference_child in zip(children, reference):
            assert isinstance(child, np.random.Generator)
            assert child.bit_generator.state != base_state
            assert same_state(child, reference_child)

    def test_integer_seed_reproduces_children_independently_of_previous_calls(self) -> None:
        n_init = 3
        nested = scripted([normal_mixture(0.0) for _ in range(2 * n_init)])
        initializer = MultiStartInitializer(nested, n_init)
        families = [normal_family()]

        initializer.initialize([0.0], families, random_state=8)
        initializer.initialize([0.0], families, random_state=7)

        repeated_children = [rng for _, _, rng in nested.calls[n_init:]]
        reference = np.random.default_rng(7).spawn(n_init)
        for child, reference_child in zip(repeated_children, reference):
            assert same_state(child, reference_child)

        fresh_nested = scripted([normal_mixture(0.0) for _ in range(n_init)])
        fresh_initializer = MultiStartInitializer(fresh_nested, n_init)

        fresh_initializer.initialize([0.0], families, random_state=7)

        fresh_children = [rng for _, _, rng in fresh_nested.calls]
        for child, reference_child in zip(fresh_children, reference):
            assert same_state(child, reference_child)

    def test_passed_source_is_used_in_place_and_advances_child_counter(self) -> None:
        n_init = 3
        nested = scripted([normal_mixture(0.0) for _ in range(2 * n_init)])
        initializer = MultiStartInitializer(nested, n_init)
        generator = np.random.default_rng(7)

        initializer.initialize([0.0], [normal_family()], random_state=generator)
        initializer.initialize([0.0], [normal_family()], random_state=generator)

        first_children = [rng for _, _, rng in nested.calls[:n_init]]
        second_children = [rng for _, _, rng in nested.calls[n_init:]]
        reference = np.random.default_rng(7).spawn(2 * n_init)[n_init:]
        for first, second, reference_child in zip(first_children, second_children, reference):
            assert not same_state(first, second)
            assert same_state(second, reference_child)


class TestFailedRuns:
    """Skipping unlucky runs, reporting them and passing other errors through."""

    def test_warns_once_when_part_is_unsuccessful(self) -> None:
        outcomes: list[Outcome] = [
            InitializationError("a"),
            normal_mixture(1.0),
            InitializationError("b"),
            normal_mixture(0.0),
        ]
        initializer = MultiStartInitializer(scripted(outcomes), len(outcomes))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = initializer.initialize([0.0], [normal_family()], random_state=7)

        assert result is outcomes[3]
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert str(caught[0].message) == "2 of 4 runs failed; the last reason: b"

    @pytest.mark.parametrize(
        "make_scenario",
        [
            pytest.param(one_failure_first, id="first-unsuccessful"),
            pytest.param(one_failure_last, id="last-unsuccessful"),
        ],
    )
    def test_success_is_found_around_single_unsuccessful_run(
        self, make_scenario: Callable[[], OutcomesAndBest]
    ) -> None:
        outcomes, best_index = make_scenario()
        initializer = MultiStartInitializer(scripted(outcomes), len(outcomes))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = initializer.initialize([0.0], [normal_family()], random_state=7)

        assert result is outcomes[best_index]
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert str(caught[0].message) == "1 of 3 runs failed; the last reason: a"

    def test_component_estimation_error_is_skipped(self) -> None:
        outcomes: list[Outcome] = [ComponentEstimationError("c"), normal_mixture(0.0)]
        initializer = MultiStartInitializer(scripted(outcomes), len(outcomes))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = initializer.initialize([0.0], [normal_family()], random_state=7)

        assert result is outcomes[1]
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert str(caught[0].message) == "1 of 2 runs failed; the last reason: c"

    @pytest.mark.filterwarnings("error")
    def test_all_unsuccessful_runs_raise_error_with_last_cause(self) -> None:
        errors: list[Outcome] = [
            ComponentEstimationError("x"),
            InitializationError("y"),
            ComponentEstimationError("z"),
        ]
        initializer = MultiStartInitializer(scripted(errors), len(errors))
        expected = "All 3 runs failed; the last reason: z"

        with pytest.raises(InitializationError, match=fullmatch(expected)) as exc_info:
            initializer.initialize([0.0], [normal_family()], random_state=7)

        assert type(exc_info.value) is InitializationError
        assert exc_info.value.__cause__ is errors[2]

    @pytest.mark.filterwarnings("error")
    def test_single_unsuccessful_run_raises_error_with_last_cause(self) -> None:
        error = InitializationError("only")
        initializer = MultiStartInitializer(scripted([error]), 1)
        expected = "All 1 runs failed; the last reason: only"

        with pytest.raises(InitializationError, match=fullmatch(expected)) as exc_info:
            initializer.initialize([0.0], [normal_family()], random_state=7)

        assert type(exc_info.value) is InitializationError
        assert exc_info.value.__cause__ is error

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize(
        "make_error",
        [
            pytest.param(lambda: ValueError("bad"), id="value-error"),
            pytest.param(lambda: TypeError("bad"), id="type-error"),
            pytest.param(lambda: NotImplementedError("bad"), id="not-implemented-error"),
            pytest.param(lambda: ImportError("bad"), id="import-error"),
        ],
    )
    def test_other_error_passes_through_and_stops_runs(self, make_error: Callable[[], Exception]) -> None:
        error = make_error()
        outcomes: list[Outcome] = [InitializationError("a"), error, normal_mixture(0.0)]
        nested = scripted(outcomes)
        initializer = MultiStartInitializer(nested, len(outcomes))

        with pytest.raises(Exception, match="^bad$") as exc_info:
            initializer.initialize([0.0], [normal_family()], random_state=7)

        assert exc_info.value is error
        assert len(nested.calls) == 2  # noqa: PLR2004

    def test_nested_coverage_failure_is_skipped(self) -> None:
        uniform = uniform_family()
        first = uniform_mixture(uniform, 0.0, 3.0)
        second = uniform_mixture(uniform, -2.0, 3.0)
        initializer = MultiStartInitializer(scripted([first, second]), 2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = initializer.initialize([-1.0, 2.0], [uniform], random_state=7)

        assert result is second
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        beginning = (
            "1 of 2 runs failed; the last reason: The log-likelihood of the starting mixture on the sample is -inf. "
        )
        assert re.fullmatch(re.escape(beginning) + r".+", str(caught[0].message))

    def test_nested_component_count_error_passes_through_and_stops_runs(self) -> None:
        uniform = uniform_family()
        wrong = MixtureModel(
            [
                uniform.distribution(lower_bound=0.0, upper_bound=3.0),
                uniform.distribution(lower_bound=0.0, upper_bound=3.0),
            ]
        )
        good = uniform_mixture(uniform, -2.0, 3.0)
        nested = scripted([wrong, good])
        initializer = MultiStartInitializer(nested, 2)

        with pytest.raises(ValueError, match="^The initializer returned 2 components for 1 families"):
            initializer.initialize([-1.0, 2.0], [uniform], random_state=7)

        assert len(nested.calls) == 1
