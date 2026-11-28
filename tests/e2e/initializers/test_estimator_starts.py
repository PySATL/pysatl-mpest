"""End-to-end scenarios: mixture estimators start from the mixtures of ``ClusterInitializer``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.types import CharacteristicName, FamilyName
from pysatl_mpest.estimators import ECM, ECME, EM
from pysatl_mpest.initializers import (
    ClusterInitializer,
    IdentityMatcher,
    InitializationError,
    KMeansPartitioner,
    Partitioner,
    QuantilePartitioner,
)
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.scenarios import (
    PARTITIONER_FACTORIES,
    PARTITIONER_IDS,
    make_ecm,
    make_ecme,
    make_em,
    make_kmeans_partitioner,
    make_quantile_partitioner,
    make_random_centers_partitioner,
    make_sklearn_agglomerative_partitioner,
    make_sklearn_kmeans_partitioner,
)

type EstimatorFactory = Callable[[], EM | ECM | ECME]

GRID = np.linspace(-20, 20, 161)


def modes_zero_and_ten() -> UnivariateFloatArray:
    """A sample from two well-separated normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 300), rng.normal(10, 1, 300)])


def two_exponential_components() -> UnivariateFloatArray:
    """A sample from two exponential components with different rates."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.exponential(1, 150), rng.exponential(20, 150)])


def two_uniform_components() -> UnivariateFloatArray:
    """A sample from two well-separated continuous uniform components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.uniform(0, 1, 150), rng.uniform(5, 6, 150)])


def normal_exponential_uniform_components() -> UnivariateFloatArray:
    """A sample from well-separated normal, exponential and continuous uniform components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 100), rng.exponential(1, 100), rng.uniform(10, 20, 100)])


def overlapping_modes() -> UnivariateFloatArray:
    """A sample from two overlapping normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 300), rng.normal(3, 1, 300)])


def uniform_and_normal() -> UnivariateFloatArray:
    """A sample from well-separated continuous uniform and normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.uniform(0, 4, 300), rng.normal(10, 1, 300)])


def make_two_normal_families() -> list[ParametricFamily]:
    """Two plain Normal families."""

    return [family(FamilyName.NORMAL) for _ in range(2)]


def make_two_exponential_families() -> list[ParametricFamily]:
    """Two plain Exponential families."""

    return [family(FamilyName.EXPONENTIAL) for _ in range(2)]


def make_two_uniform_families() -> list[ParametricFamily]:
    """Two plain ContinuousUniform families."""

    return [family(FamilyName.CONTINUOUS_UNIFORM) for _ in range(2)]


def make_normal_exponential_uniform_families() -> list[ParametricFamily]:
    """Plain Normal, Exponential and ContinuousUniform families."""

    return [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL), family(FamilyName.CONTINUOUS_UNIFORM)]


def make_em_one_step() -> EM:
    """A one-iteration EM estimator."""

    return make_em(1)


def make_ecm_one_step() -> ECM:
    """A one-iteration ECM estimator."""

    return make_ecm(1)


def make_ecme_one_step() -> ECME:
    """A one-iteration ECME estimator."""

    return make_ecme(1)


def make_ecm_two_steps() -> ECM:
    """A two-iteration ECM estimator."""

    return make_ecm(2)


def make_ecme_two_steps() -> ECME:
    """A two-iteration ECME estimator."""

    return make_ecme(2)


def make_em_three_steps() -> EM:
    """A three-iteration EM estimator."""

    return make_em(3)


def make_ecm_three_steps() -> ECM:
    """A three-iteration ECM estimator."""

    return make_ecm(3)


def make_ecme_three_steps() -> ECME:
    """A three-iteration ECME estimator."""

    return make_ecme(3)


def make_em_twenty_steps() -> EM:
    """A twenty-iteration EM estimator."""

    return make_em(20)


def make_ecm_twenty_steps() -> ECM:
    """A twenty-iteration ECM estimator."""

    return make_ecm(20)


def is_from(component: ParametricFamilyDistribution, family_: ParametricFamily) -> bool:
    """Whether the component has the name and the default parametrization of the family."""

    return component.family_name == family_.name and component.parametrization_name == family_.base_parametrization_name


def assert_fixed(component: ParametricFamilyDistribution, family_: ParametricFamily, **fixed: float) -> None:
    """Check that the fixed values are not free and the component is the base family with them."""

    assert not set(fixed) & set(component.parameters)
    expected = family_(**fixed, **component.parameters)
    density = np.asarray(component.calculate_characteristic(CharacteristicName.PDF, GRID))
    expected_density = np.asarray(expected.calculate_characteristic(CharacteristicName.PDF, GRID))
    assert density == pytest.approx(expected_density, rel=1e-12)


@pytest.mark.parametrize("make_partitioner", PARTITIONER_FACTORIES, ids=PARTITIONER_IDS)
@pytest.mark.parametrize(
    ("make_sample", "make_families"),
    [
        pytest.param(modes_zero_and_ten, make_two_normal_families, id="modes-zero-and-ten-two-normals"),
        pytest.param(two_exponential_components, make_two_exponential_families, id="two-exponentials"),
        pytest.param(two_uniform_components, make_two_uniform_families, id="two-uniforms"),
        pytest.param(
            normal_exponential_uniform_components,
            make_normal_exponential_uniform_families,
            id="normal-exponential-uniform",
        ),
    ],
)
@pytest.mark.filterwarnings("error")
def test_em_starts_from_every_partitioner_and_keeps_families(
    make_partitioner: Callable[[], Partitioner],
    make_sample: Callable[[], UnivariateFloatArray],
    make_families: Callable[[], list[ParametricFamily]],
) -> None:
    """EM starts from the cluster mixture and returns finite components of the given families."""

    X = make_sample()
    families = make_families()
    start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

    result = make_em(2).fit(X, start)

    assert result.n_components == len(families)
    assert all(is_from(result[j], families[j]) for j in range(len(families)))
    assert np.isfinite(result.loglikelihood(X))


@pytest.mark.parametrize("make_partitioner", PARTITIONER_FACTORIES, ids=PARTITIONER_IDS)
@pytest.mark.parametrize(
    "make_estimator",
    [pytest.param(make_ecm_two_steps, id="ecm"), pytest.param(make_ecme_two_steps, id="ecme")],
)
@pytest.mark.filterwarnings("error")
def test_conditional_estimators_start_on_three_family_sample(
    make_partitioner: Callable[[], Partitioner], make_estimator: EstimatorFactory
) -> None:
    """ECM and ECME start from the cluster mixture on a sample with three different families."""

    X = normal_exponential_uniform_components()
    families = make_normal_exponential_uniform_families()
    start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

    result = make_estimator().fit(X, start)

    assert result.n_components == len(families)
    assert all(is_from(result[j], families[j]) for j in range(len(families)))
    assert np.isfinite(result.loglikelihood(X))


@pytest.mark.parametrize(
    "make_estimator",
    [
        pytest.param(make_em_one_step, id="em"),
        pytest.param(make_ecm_one_step, id="ecm"),
        pytest.param(make_ecme_one_step, id="ecme"),
    ],
)
@pytest.mark.filterwarnings("error")
def test_point_outside_its_own_component_is_handed_over_by_one_iteration(make_estimator: EstimatorFactory) -> None:
    """One iteration strictly improves a finite start whose exponential cluster contains a negative point."""

    X = np.array([-1, 5, 6, 7, 20, 21, 22, 23], dtype=np.float64)
    families = [family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)]
    start = ClusterInitializer(QuantilePartitioner(), matcher=IdentityMatcher()).initialize(
        X, families, random_state=42
    )

    result = make_estimator().fit(X, start)

    assert np.isfinite(result.loglikelihood(X))
    assert result.loglikelihood(X) > start.loglikelihood(X)


@pytest.mark.parametrize(
    ("make_partitioner", "make_estimator"),
    [
        pytest.param(make_kmeans_partitioner, make_em_twenty_steps, id="kmeans-em"),
        pytest.param(make_quantile_partitioner, make_em_twenty_steps, id="quantile-em"),
        pytest.param(make_random_centers_partitioner, make_em_twenty_steps, id="random-centers-em"),
        pytest.param(make_sklearn_kmeans_partitioner, make_em_twenty_steps, id="sklearn-kmeans-em"),
        pytest.param(make_sklearn_agglomerative_partitioner, make_em_twenty_steps, id="sklearn-agglomerative-em"),
        pytest.param(make_kmeans_partitioner, make_ecm_twenty_steps, id="kmeans-ecm"),
        pytest.param(make_kmeans_partitioner, make_ecme_three_steps, id="kmeans-ecme"),
    ],
)
@pytest.mark.filterwarnings("error")
def test_fixed_mean_survives_estimator_while_free_parameters_move(
    make_partitioner: Callable[[], Partitioner], make_estimator: EstimatorFactory
) -> None:
    """A fixed mean is unchanged while the scale and the free mean improve the likelihood."""

    X = overlapping_modes()
    normal = family(FamilyName.NORMAL)
    families = [normal.view(mu=0), normal]
    start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

    result = make_estimator().fit(X, start)

    assert_fixed(result[0], normal, mu=0)
    assert result[0].parameters["sigma"] != start[0].parameters["sigma"]
    assert result.loglikelihood(X) > start.loglikelihood(X)
    assert result[1].parameters["mu"] == pytest.approx(3, abs=0.2)


@pytest.mark.parametrize(
    "make_estimator",
    [
        pytest.param(make_em_three_steps, id="em"),
        pytest.param(make_ecm_three_steps, id="ecm"),
        pytest.param(make_ecme_three_steps, id="ecme"),
    ],
)
@pytest.mark.filterwarnings("error")
def test_fixed_scale_survives_start_and_estimator(make_estimator: EstimatorFactory) -> None:
    """A fixed scale is exact before and after fitting while the free mean reaches the right mode."""

    X = modes_zero_and_ten()
    normal = family(FamilyName.NORMAL)
    families = [normal, normal.view(sigma=1)]
    start = ClusterInitializer(KMeansPartitioner()).initialize(X, families, random_state=42)

    result = make_estimator().fit(X, start)

    assert_fixed(start[1], normal, sigma=1)
    assert_fixed(result[1], normal, sigma=1)
    assert result[1].parameters["mu"] == pytest.approx(10, abs=0.2)


@pytest.mark.parametrize("make_partitioner", PARTITIONER_FACTORIES, ids=PARTITIONER_IDS)
@pytest.mark.filterwarnings("error")
def test_component_order_follows_families_not_cluster_means(make_partitioner: Callable[[], Partitioner]) -> None:
    """Fixed means keep the family order even when it is opposite to the order of cluster means."""

    X = modes_zero_and_ten()
    normal = family(FamilyName.NORMAL)
    families = [normal.view(mu=10), normal.view(mu=0)]
    start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

    result = make_em(5).fit(X, start)

    for mixture in (start, result):
        assert_fixed(mixture[0], normal, mu=10)
        assert_fixed(mixture[1], normal, mu=0)


@pytest.mark.parametrize(
    "make_partitioner",
    [
        pytest.param(make_kmeans_partitioner, id="kmeans"),
        pytest.param(make_quantile_partitioner, id="quantile"),
        pytest.param(make_random_centers_partitioner, id="random-centers"),
    ],
)
@pytest.mark.filterwarnings("error")
def test_fixed_lower_bound_survives_and_upper_bound_is_fitted(make_partitioner: Callable[[], Partitioner]) -> None:
    """A fixed lower bound is exact while the free upper bound is fitted near the true value."""

    X = uniform_and_normal()
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)
    families = [uniform.view(lower_bound=0), family(FamilyName.NORMAL)]
    start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

    result = make_em(3).fit(X, start)

    assert_fixed(start[0], uniform, lower_bound=0)
    assert_fixed(result[0], uniform, lower_bound=0)
    assert result[0].parameters["upper_bound"] == pytest.approx(4, abs=0.1)


@pytest.mark.filterwarnings("error")
def test_fixed_value_contradicted_by_data_is_a_failed_run() -> None:
    """A fixed bound contradicted by the data fails the run instead of corrupting the start."""

    X = uniform_and_normal()
    families = [family(FamilyName.CONTINUOUS_UNIFORM).view(upper_bound=3), family(FamilyName.NORMAL)]

    with pytest.raises(InitializationError, match=r"^No assignment gives every family a suitable cluster\.") as error:
        ClusterInitializer(KMeansPartitioner()).initialize(X, families, random_state=42)

    assert "outside the support" in str(error.value)
