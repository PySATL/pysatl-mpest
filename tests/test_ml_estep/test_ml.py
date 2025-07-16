import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sklearn.cluster import KMeans

from mpest import Distribution, MixtureDistribution, Problem
from mpest.em.methods.likelihood_method import ClusteringEStep
from mpest.models import GaussianModel, WeibullModelExp
from mpest.utils import ResultWithError

WEIBULL_PARAMS_COUNT = 2
MIN_COMPONENT_SIZE = 10


def valid_weibull_data():
    return st.lists(
        st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10, max_size=1000
    ).map(np.array)


def valid_gaussian_data():
    return st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10, max_size=1000
    ).map(np.array)


def mixed_data():
    return st.one_of(valid_weibull_data(), valid_gaussian_data())


def mixture_problems():
    return st.builds(
        lambda samples: Problem(
            samples,
            MixtureDistribution.from_distributions([
                Distribution.from_params(WeibullModelExp, [1.0, 1.0]),
                Distribution.from_params(GaussianModel, [0.0, 1.0]),
            ], [0.5, 0.5])
        ),
        mixed_data()
    )


class TestClusteringEStepInitialization:
    @given(st.lists(st.sampled_from([WeibullModelExp(), GaussianModel()])),
                    st.integers(min_value=0, max_value=1000))
    def test_initialization(self, models, labels_seed):
        assume(len(models) > 0)
        np.random.seed(labels_seed)
        labels = np.random.randint(0, len(models), size=100)
        ml = ClusteringEStep(models, labels)
        assert ml._n_components == len(models)
        assert len(ml._models) == len(models)
        assert not ml._initialized
        assert ml._current_mixture.distributions == []


class TestWeibullParamEstimation:

    @given(valid_weibull_data())
    def test_weibull_param_estimation(self, data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ClusteringEStep(models, np.zeros(len(data), dtype=int))
        params = ml._estimate_weibull_params(data)
        assert len(params) == WEIBULL_PARAMS_COUNT
        assert params[0] > 0
        assert params[1] > 0

    @given(st.lists(st.floats(min_value=0, max_value=0, allow_nan=False), min_size=1))
    def test_weibull_param_estimation_with_bad_data(self, data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ClusteringEStep(models, np.zeros(len(data), dtype=int))
        params = ml._estimate_weibull_params(np.array(data))
        assert params[0] > 0
        assert isinstance(params[1], float)


class TestDistributionInitialization:
    @given(mixed_data(), st.booleans())
    def test_initialization(self, data, accurate_init):
        assume(len(data) >= MIN_COMPONENT_SIZE)
        models = [WeibullModelExp(), GaussianModel()]
        labels = np.random.randint(0, len(models), size=len(data))
        ml = ClusteringEStep(models, labels, accurate_init=accurate_init)
        mixture = ml._initialize_distributions(data, labels)
        assert len(mixture.distributions) == len(models)
        for dist in mixture.distributions:
            assert dist.params is not None
            if isinstance(dist.model, WeibullModelExp):
                assert dist.params[0] > 0
                assert dist.params[1] > 0
            else:
                assert dist.params[1] > 0


class TestEStep:
    @given(mixture_problems())
    def test_e_step(self, problem):
        models = [WeibullModelExp(), GaussianModel()]
        clusterizer = KMeans(n_clusters=len(models))
        ml = ClusteringEStep(models, clusterizer)
        result = ml.step(problem)

        if isinstance(result, ResultWithError):
            pytest.fail("Unexpected error in E-step")
        else:
            new_problem, h = result
            assert len(new_problem.samples) <= len(problem.samples)
            assert h.shape[0] == len(models)
            if len(new_problem.samples) > 0:
                assert h.shape[1] == len(new_problem.samples)
                for col in h.T:
                    assert pytest.approx(1.0, abs=1e-6) == sum(col)

    @given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False)),
                    st.integers(min_value=1, max_value=10))
    def test_e_step_with_empty_cluster(self, data, n_components):
        data = np.array(data)
        assume(len(data) >= n_components)
        models = [WeibullModelExp() if i % 2 else GaussianModel() for i in range(n_components)]
        clusterizer = KMeans(n_clusters=len(models))
        ml = ClusteringEStep(models, clusterizer)

        initial_mixture = MixtureDistribution.from_distributions(
            [Distribution.from_params(model.__class__, [1.0, 1.0]) for model in models]
        )
        problem = Problem(data, initial_mixture)

        result = ml.step(problem)
        if isinstance(result, ResultWithError):
            pytest.fail("Unexpected error in E-step")
        else:
            new_problem, h = result
            assert h.shape == (n_components, len(new_problem.samples))


class TestEdgeCases:
    @given(mixed_data())
    def test_single_component(self, data):
        assume(len(data) >= MIN_COMPONENT_SIZE)
        models = [WeibullModelExp()]
        labels = np.zeros(len(data), dtype=int)
        ml = ClusteringEStep(models, labels)
        mixture = ml._initialize_distributions(data, labels)
        assert len(mixture.distributions) == 1
        assert mixture.distributions[0].params[0] > 0
        assert mixture.distributions[0].params[1] > 0
