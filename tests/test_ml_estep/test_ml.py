import numpy as np
import pytest
from scipy.stats import weibull_min, norm

from mpest import Problem, MixtureDistribution, Distribution
from mpest.em.methods.likelihood_method import ML
from mpest.models import WeibullModelExp, GaussianModel


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
        ml = ML(models, n_components=2, method="kmeans")
        assert ml._n_components == 2
        assert ml._method == "kmeans"
        assert len(ml._models) == 2
        assert not ml._initialized
        assert ml._current_mixture is None

    def test_initialization_with_invalid_method(self):
        models = [WeibullModelExp(), GaussianModel()]
        with pytest.raises(ValueError):
            ML(models, n_components=2, method="invalid_method")._initialize_distributions(np.ndarray([]))


class TestWeibullParamEstimation:
    def test_weibull_param_estimation(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2)

        data = weibull_min.rvs(1.5, scale=2.0, size=1000)
        params = ml._estimate_weibull_params(data)
        assert len(params) == 2
        assert params[0] > 0
        assert params[1] > 0

    def test_weibull_param_estimation_with_bad_data(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2)

        data = np.array([0, 0, 0])
        params = ml._estimate_weibull_params(data)
        assert params[0] > 0
        assert isinstance(params[1], float)


class TestDistributionInitialization:
    def test_kmeans_initialization(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2, method="kmeans")

        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == 2

    def test_dbscan_initialization(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2, method="dbscan")

        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) > 0

    def test_agglo_initialization(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2, method="agglo")

        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == 2


class TestEStep:
    def test_e_step(self, sample_data):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2, method="kmeans")

        initial_mixture = MixtureDistribution.from_distributions(
            [
                Distribution.from_params(WeibullModelExp, [1.0, 1.0]),
                Distribution.from_params(GaussianModel, [0.0, 1.0]),
            ],
            [0.5, 0.5]
        )
        problem = Problem(sample_data, initial_mixture)

        result = ml.step(problem)
        assert len(result) == 3

        active_samples, h, _ = result
        assert len(active_samples) == len(sample_data)
        assert h.shape == (2, len(sample_data))

        for i in range(h.shape[1]):
            assert pytest.approx(1.0) == sum(h[:, i])

    def test_e_step_with_empty_cluster(self):
        data = np.concatenate([np.zeros(500), np.ones(500)])
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2, method="kmeans")

        initial_mixture = MixtureDistribution.from_distributions(
            [
                Distribution.from_params(WeibullModelExp, [1.0, 1.0]),
                Distribution.from_params(GaussianModel, [0.0, 1.0]),
            ],
            [0.5, 0.5]
        )
        problem = Problem(data, initial_mixture)

        result = ml.step(problem)
        active_samples, h, _ = result
        assert len(active_samples) == len(data)
        assert h.shape == (2, len(data))



class TestEdgeCases:
    def test_empty_input(self):
        models = [WeibullModelExp(), GaussianModel()]
        ml = ML(models, n_components=2)
        with pytest.raises(ValueError):
            ml._initialize_distributions(np.array([]))

    def test_single_component(self, sample_data):
        models = [WeibullModelExp()]
        ml = ML(models, n_components=1)
        mixture = ml._initialize_distributions(sample_data)
        assert len(mixture.distributions) == 1
