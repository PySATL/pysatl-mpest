"""Unit tests for ``RefiningInitializer`` with scripted roles."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import BaseEstimator
from pysatl_mpest.initializers.exceptions import ComponentEstimationError, InitializationError
from pysatl_mpest.initializers.refining import RefiningInitializer
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.estimator_data import exponential as exponential_component
from tests.helpers.estimator_data import normal as normal_component
from tests.helpers.families import family
from tests.mocks.initializers import ScriptedInitializer

X_INPUT = [0.0, 1.0, 2.0]

type Result = Callable[[MixtureModel], MixtureModel]


class RecordingEstimator(BaseEstimator):
    """Estimator that records its arguments and returns a scripted result."""

    def __init__(self, result: Result | None = None) -> None:
        self._result = result if result is not None else lambda mixture: mixture
        self.calls: list[tuple[ArrayLike, MixtureModel, dict[str, Any]]] = []

    def fit(self, X: ArrayLike, mixture: MixtureModel, tolerance: float = 1e-3, *, groups: Any = None) -> MixtureModel:
        self.calls.append((X, mixture, {"tolerance": tolerance, "groups": groups}))
        return self._result(mixture)


class OptionsEstimator(BaseEstimator):
    """Estimator whose ``fit`` accepts arbitrary keyword arguments."""

    def fit(self, X: ArrayLike, mixture: MixtureModel, **options: Any) -> MixtureModel:
        return mixture


class RequiredEstimator(BaseEstimator):
    """Estimator whose ``fit`` requires one more positional argument."""

    def fit(self, X: ArrayLike, mixture: MixtureModel, alpha: float) -> MixtureModel:
        return mixture


class FitOnly:
    """Class with a ``fit`` method that is not an estimator."""

    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        return mixture


def normal_family() -> ParametricFamily:
    """The registered normal family."""

    return family(FamilyName.NORMAL)


def uniform_family() -> ParametricFamily:
    """The registered continuous uniform family."""

    return family(FamilyName.CONTINUOUS_UNIFORM)


def three_normal_families() -> list[ParametricFamily]:
    """Three references to the normal family."""

    return [normal_family()] * 3


def normal_mixture(mus: Sequence[float]) -> MixtureModel:
    """A normal mixture with unit standard deviations."""

    return MixtureModel([normal_component(mu, 1.0) for mu in mus])


def start_mixture() -> MixtureModel:
    """The default three-component start."""

    return normal_mixture([0.0, 1.0, 2.0])


def scripted_result(mixture: MixtureModel) -> ScriptedInitializer:
    """A nested initializer that always returns the mixture."""

    def build(X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator) -> MixtureModel:
        return mixture

    return ScriptedInitializer(build)


def refining(
    nested: ScriptedInitializer | None = None,
    estimator: BaseEstimator | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> RefiningInitializer:
    """A refining initializer with default test roles."""

    return RefiningInitializer(
        nested if nested is not None else scripted_result(start_mixture()),
        estimator if estimator is not None else RecordingEstimator(),
        fit_kwargs=fit_kwargs,
    )


def fullmatch(text: str) -> str:
    """Regex that matches the whole text literally."""

    return f"^{re.escape(text)}$"


class TestConstruction:
    """Storage and copying of the constructor arguments."""

    def test_stores_roles_as_same_objects_and_empty_fit_kwargs_by_default(self) -> None:
        nested = scripted_result(start_mixture())
        estimator = RecordingEstimator()

        initializer = RefiningInitializer(nested, estimator)

        assert initializer.initializer is nested
        assert initializer.estimator is estimator
        assert initializer.fit_kwargs == {}
        assert type(initializer.fit_kwargs) is dict

    def test_default_fit_kwargs_are_not_shared_between_objects(self) -> None:
        first = refining()
        second = refining()

        assert first.fit_kwargs is not second.fit_kwargs

    @pytest.mark.parametrize(
        "make_fit_kwargs",
        [
            pytest.param(lambda: {"tolerance": 0.5}, id="dict"),
            pytest.param(lambda: MappingProxyType({"tolerance": 0.5}), id="mapping-proxy"),
        ],
    )
    def test_copies_fit_kwargs_and_later_mapping_changes_do_not_reach_fit(
        self, make_fit_kwargs: Callable[[], Mapping[str, Any]]
    ) -> None:
        user_kwargs = make_fit_kwargs()
        estimator = RecordingEstimator()
        initializer = refining(estimator=estimator, fit_kwargs=user_kwargs)

        assert initializer.fit_kwargs == user_kwargs
        assert type(initializer.fit_kwargs) is dict
        assert initializer.fit_kwargs is not user_kwargs

        if isinstance(user_kwargs, dict):
            user_kwargs["groups"] = {0: 1}

            assert "groups" not in initializer.fit_kwargs

            initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

            assert estimator.calls[0][2] == {"tolerance": 0.5, "groups": None}

    @pytest.mark.parametrize(
        ("make_bad_initializer", "type_name"),
        [
            pytest.param(lambda: ScriptedInitializer, "ABCMeta", id="initializer-class"),
            pytest.param(lambda: "initializer", "str", id="string"),
        ],
    )
    def test_rejects_initializer_of_wrong_type(
        self, make_bad_initializer: Callable[[], object], type_name: str
    ) -> None:
        expected = f"initializer must be an Initializer, got {type_name}."

        with pytest.raises(TypeError, match=fullmatch(expected)):
            RefiningInitializer(make_bad_initializer(), RecordingEstimator())

    @pytest.mark.parametrize(
        ("make_bad_estimator", "type_name"),
        [
            pytest.param(lambda: RecordingEstimator, "ABCMeta", id="estimator-class"),
            pytest.param(lambda: FitOnly(), "FitOnly", id="fit-only-object"),
            pytest.param(lambda: "estimator", "str", id="string"),
        ],
    )
    def test_rejects_estimator_of_wrong_type(self, make_bad_estimator: Callable[[], object], type_name: str) -> None:
        expected = f"estimator must be a BaseEstimator, got {type_name}."

        with pytest.raises(TypeError, match=fullmatch(expected)):
            RefiningInitializer(scripted_result(start_mixture()), make_bad_estimator())

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([("tolerance", 0.5)], id="list-of-pairs"),
            pytest.param("tolerance", id="string"),
            pytest.param(5, id="integer"),
        ],
    )
    def test_rejects_fit_kwargs_of_wrong_type(self, value: object) -> None:
        expected = f"fit_kwargs must be a mapping or None, got {type(value).__name__}."

        with pytest.raises(TypeError, match=fullmatch(expected)):
            refining(fit_kwargs=value)


class TestFitKwargs:
    """Static checking of the keyword arguments against the signature of ``fit``."""

    @pytest.mark.parametrize(
        ("make_estimator", "make_fit_kwargs"),
        [
            pytest.param(lambda: RecordingEstimator(), lambda: {"tolerance": 0.5}, id="tolerance"),
            pytest.param(lambda: RecordingEstimator(), lambda: {"groups": {0: 1}}, id="groups"),
            pytest.param(
                lambda: RecordingEstimator(),
                lambda: {"tolerance": 0.5, "groups": {0: 1}},
                id="tolerance-and-groups",
            ),
            pytest.param(lambda: OptionsEstimator(), lambda: {"anything": 1}, id="arbitrary-option"),
        ],
    )
    def test_accepts_names_from_fit_signature(
        self, make_estimator: Callable[[], BaseEstimator], make_fit_kwargs: Callable[[], dict[Any, Any]]
    ) -> None:
        fit_kwargs = make_fit_kwargs()

        initializer = refining(estimator=make_estimator(), fit_kwargs=fit_kwargs)

        assert initializer.fit_kwargs == fit_kwargs

    def test_rejects_unknown_name_with_signature_error_as_cause(self) -> None:
        with pytest.raises(
            TypeError, match=r"^fit_kwargs do not match the signature of RecordingEstimator\.fit: .*\.$"
        ) as exc_info:
            refining(fit_kwargs={"tol": 0.5})

        error = exc_info.value
        expected = f"fit_kwargs do not match the signature of RecordingEstimator.fit: {error.__cause__!s}."
        assert str(error) == expected
        assert type(error.__cause__) is TypeError

    @pytest.mark.parametrize(
        ("make_estimator", "name"),
        [
            pytest.param(lambda: RecordingEstimator(), "X", id="recording-X"),
            pytest.param(lambda: RecordingEstimator(), "mixture", id="recording-mixture"),
            pytest.param(lambda: OptionsEstimator(), "X", id="options-X"),
            pytest.param(lambda: OptionsEstimator(), "mixture", id="options-mixture"),
        ],
    )
    def test_rejects_positional_fit_arguments_in_fit_kwargs(
        self, make_estimator: Callable[[], BaseEstimator], name: str
    ) -> None:
        with pytest.raises(TypeError, match="^fit_kwargs do not match the signature of"):
            refining(estimator=make_estimator(), fit_kwargs={name: 1})

    def test_rejects_missing_required_fit_argument(self) -> None:
        with pytest.raises(TypeError, match="^fit_kwargs do not match the signature of"):
            refining(estimator=RequiredEstimator())

    def test_accepts_required_fit_argument(self) -> None:
        initializer = refining(estimator=RequiredEstimator(), fit_kwargs={"alpha": 1})

        assert initializer.fit_kwargs == {"alpha": 1}

    def test_rejects_non_string_key(self) -> None:
        with pytest.raises(TypeError, match="^fit_kwargs do not match the signature of"):
            refining(estimator=OptionsEstimator(), fit_kwargs={1: 2})

    def test_creation_calls_neither_fit_nor_nested_initializer(self) -> None:
        nested = scripted_result(start_mixture())
        estimator = RecordingEstimator()

        refining(nested, estimator, fit_kwargs={"tolerance": 0.5})

        assert nested.calls == []
        assert estimator.calls == []


class TestRefinement:
    """Order and arguments of the nested initialization and the refinement."""

    def test_nested_initializer_receives_checked_input_same_families_and_same_generator(self) -> None:
        nested = scripted_result(start_mixture())
        initializer = refining(nested)
        families = three_normal_families()
        generator = np.random.default_rng(7)

        initializer.initialize(X_INPUT, families, random_state=generator)

        assert len(nested.calls) == 1
        received_X, received_families, received_rng = nested.calls[0]
        assert received_X.dtype == np.float64
        assert not received_X.flags.writeable
        np.testing.assert_array_equal(received_X, X_INPUT)
        assert received_families is families
        assert received_rng is generator

    def test_fit_receives_sample_and_start_positionally_after_nested_initializer(self) -> None:
        start = start_mixture()
        estimator = RecordingEstimator()
        fit_calls_at_nested_call = []

        def build(
            X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
        ) -> MixtureModel:
            fit_calls_at_nested_call.append(len(estimator.calls))
            return start

        nested = ScriptedInitializer(build)
        initializer = refining(nested, estimator, fit_kwargs={"tolerance": 0.5})

        initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert fit_calls_at_nested_call == [0]
        assert len(estimator.calls) == 1
        received_X, received_mixture, received_options = estimator.calls[0]
        received_sample = np.asarray(received_X)
        assert received_sample.dtype == np.float64
        np.testing.assert_array_equal(received_sample, X_INPUT)
        assert received_mixture is start
        assert received_options == {"tolerance": 0.5, "groups": None}

    def test_returns_same_object_returned_by_fit(self) -> None:
        replacement = start_mixture()
        estimator = RecordingEstimator(lambda mixture: replacement)
        initializer = refining(estimator=estimator)

        result = initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert result is replacement

    def test_applies_same_fit_kwargs_on_every_call_without_changing_field(self) -> None:
        estimator = RecordingEstimator()
        fit_kwargs = {"tolerance": 0.5}
        initializer = refining(estimator=estimator, fit_kwargs=fit_kwargs)
        before = dict(initializer.fit_kwargs)

        initializer.initialize(X_INPUT, three_normal_families(), random_state=7)
        initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert len(estimator.calls) == 2  # noqa: PLR2004
        for _, _, received_options in estimator.calls:
            assert received_options == {"tolerance": 0.5, "groups": None}
        assert initializer.fit_kwargs == before


class TestErrors:
    """Failures of the refinement and errors that pass through."""

    @pytest.mark.parametrize(
        "returned_count",
        [
            pytest.param(2, id="two-of-three"),
            pytest.param(1, id="one-of-three"),
        ],
    )
    def test_rejects_result_with_fewer_components_as_failed_run(self, returned_count: int) -> None:
        def result(mixture: MixtureModel) -> MixtureModel:
            return MixtureModel(list(mixture.components[:returned_count]))

        initializer = refining(estimator=RecordingEstimator(result))
        expected = f"The estimator returned {returned_count} of 3 components."

        with pytest.raises(InitializationError, match=fullmatch(expected)) as exc_info:
            initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert type(exc_info.value) is InitializationError

    def test_result_with_exact_component_count_passes(self) -> None:
        replacement = start_mixture()
        initializer = refining(estimator=RecordingEstimator(lambda mixture: replacement))

        result = initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert result is replacement

    def test_result_with_extra_component_is_rejected_by_common_check(self) -> None:
        replacement = normal_mixture([0.0, 1.0, 2.0, 3.0])
        initializer = refining(estimator=RecordingEstimator(lambda mixture: replacement))

        with pytest.raises(ValueError, match="^The initializer returned 4 components for 3 families"):
            initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

    @pytest.mark.parametrize(
        "make_error",
        [
            pytest.param(lambda: InitializationError("nested"), id="initialization-error"),
            pytest.param(lambda: ComponentEstimationError("nested"), id="component-estimation-error"),
        ],
    )
    def test_nested_initialization_error_passes_through_without_fit(
        self, make_error: Callable[[], InitializationError]
    ) -> None:
        error = make_error()

        def build(
            X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
        ) -> MixtureModel:
            raise error

        estimator = RecordingEstimator()
        initializer = refining(ScriptedInitializer(build), estimator)

        with pytest.raises(InitializationError, match="^nested$") as exc_info:
            initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert exc_info.value is error
        assert estimator.calls == []

    @pytest.mark.parametrize(
        "make_error",
        [
            pytest.param(lambda: ValueError("bad"), id="value-error"),
            pytest.param(lambda: RuntimeError("bad"), id="runtime-error"),
            pytest.param(lambda: InitializationError("bad"), id="initialization-error"),
        ],
    )
    def test_fit_error_passes_through_as_same_object(self, make_error: Callable[[], Exception]) -> None:
        error = make_error()

        def result(mixture: MixtureModel) -> MixtureModel:
            raise error

        initializer = refining(estimator=RecordingEstimator(result))

        with pytest.raises(Exception, match="^bad$") as exc_info:
            initializer.initialize(X_INPUT, three_normal_families(), random_state=7)

        assert exc_info.value is error

    def test_common_coverage_check_applies_to_refined_mixture(self) -> None:
        uniform = uniform_family()
        start = MixtureModel([uniform.distribution(lower_bound=-1.0, upper_bound=3.0)])
        replacement = MixtureModel([uniform.distribution(lower_bound=5.0, upper_bound=6.0)])
        initializer = RefiningInitializer(
            scripted_result(start),
            RecordingEstimator(lambda mixture: replacement),
        )

        with pytest.raises(
            InitializationError, match="^The log-likelihood of the starting mixture on the sample is -inf"
        ):
            initializer.initialize(X_INPUT, [uniform], random_state=7)

    def test_common_family_check_applies_to_refined_mixture(self) -> None:
        replacement = MixtureModel([exponential_component(1.0), normal_component(1.0, 1.0), normal_component(2.0, 1.0)])
        initializer = refining(estimator=RecordingEstimator(lambda mixture: replacement))

        with pytest.raises(ValueError, match=r"^Component 0 of the result is not from families\[0\]"):
            initializer.initialize(X_INPUT, three_normal_families(), random_state=7)
