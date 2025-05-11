import numpy as np
import pytest
from scipy.stats import norm, weibull_min

from mpest import Distribution, MixtureDistribution, Problem
from mpest.em.methods.likelihood_method import ML
from mpest.models import GaussianModel, WeibullModelExp


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    data1 = weibull_min.rvs(1.5, scale=1.0, size=n_samples // 2)
    data2 = norm.rvs(loc=5, scale=1.0, size=n_samples // 2)
    return np.concatenate([data1, data2])


class TestMLInitialization:
    def test_initialization(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, method="kmeans")
        assert ml._n_components == len(models)
        assert ml._method == "kmeans"
        assert len(ml._models) == len(models)
        assert not ml._initialized
        assert ml._current_mixture.distributions == []

    def test_initialization_with_invalid_method(self):
        models = [WeibullModelExp(), GaussianModel()]
        with pytest.raises(ValueError):
            ML(models, method="invalid_method")._initialize_distributions(np.ndarray([]))


class TestWeibullParamEstimation:
    def test_weibull_param_estimation(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models)
        count_of_params = 2
        data = weibull_min.rvs(1.5, scale=2.0, size=1000)
        params = ml.estimate_weibull_params(data)
        assert len(params) == count_of_params
        assert params[0] > 0
        assert params[1] > 0

    def test_weibull_param_estimation_with_bad_data(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models)

        data = np.array([0, 0, 0])
        params = ml.estimate_weibull_params(data)
        assert params[0] > 0
        assert isinstance(params[1], float)


class TestDistributionInitialization:
    def test_kmeans_initialization(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, method="kmeans")

        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == len(models)

    def test_dbscan_initialization(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, method="dbscan")

        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == len(models)

    def test_agglo_initialization(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, method="agglo")

        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == len(models)


class TestEStep:
    def test_e_step(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, method="kmeans")

        initial_mixture = MixtureDistribution.from_distributions(
            [
                Distribution.from_params(WeibullModelExp, [1.0, 1.0]),
                Distribution.from_params(GaussianModel, [0.0, 1.0]),
            ],
            [0.5, 0.5],
        )
        problem = Problem(sample_data, initial_mixture)

        result = ml.step(problem)

        active_samples, h, _ = result
        assert len(active_samples) == len(sample_data)
        assert h.shape == (2, len(sample_data))

        for i in range(h.shape[1]):
            assert pytest.approx(1.0) == sum(h[:, i])

    def test_e_step_with_empty_cluster(self):
        data = np.concatenate([np.zeros(500), np.ones(500)])
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, method="kmeans")

        initial_mixture = MixtureDistribution.from_distributions(
            [
                Distribution.from_params(WeibullModelExp, [1.0, 1.0]),
                Distribution.from_params(GaussianModel, [0.0, 1.0]),
            ],
            [0.5, 0.5],
        )
        problem = Problem(data, initial_mixture)

        result = ml.step(problem)
        active_samples, h, _ = result
        assert len(active_samples) == len(data)
        assert h.shape == (2, len(data))


class TestEdgeCases:
    def test_empty_input(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models)
        with pytest.raises(ValueError):
            ml._initialize_distributions(np.array([]))

    def test_single_component(self, sample_data):
        models = [WeibullModelExp()]
        ml = ML(models)
        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == 1
