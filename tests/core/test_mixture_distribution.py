from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mpest import Distribution
from mpest.core.mixture_distribution import DistributionInMixture, MixtureDistribution
from mpest.models import AModel, AModelWithGenerator


def valid_size():
    return st.integers(min_value=1, max_value=100)


def valid_x():
    return st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)


def valid_params():
    return st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=5
    ).map(np.array)


def valid_prior_probability():
    return st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, exclude_min=False, exclude_max=False))


def valid_distributions():
    return st.lists(st.builds(Distribution, model=st.just(MockModel()), params=valid_params()), min_size=1, max_size=5)


class MockModel(AModel):
    @property
    def name(self):
        return "MockModel"

    def pdf(self, x, params):
        return x * params[0]

    def lpdf(self, x, params):
        return np.log(self.pdf(x, params))

    def params_convert_from_model(self, params):
        return params

    def params_convert_to_model(self, params):
        return params


class MockModelWithGenerator(AModelWithGenerator):
    @property
    def name(self):
        return "MockModelWithGenerator"

    def pdf(self, x, params):
        return x * params[0]

    def lpdf(self, x, params):
        return np.log(self.pdf(x, params))

    def params_convert_from_model(self, params):
        return params

    def params_convert_to_model(self, params):
        return params

    def generate(self, params, normalized=True, size=1):
        return np.random.uniform(0, 1, size=size)


class TestDistributionInMixture:
    @given(valid_params(), valid_prior_probability())
    def test_init(self, params, prior_probability):
        model = MagicMock(spec=AModel)
        dist = DistributionInMixture(model=model, params=params, prior_probability=prior_probability)

        assert dist.params is params
        assert dist.model is model
        assert dist.prior_probability == prior_probability

    @given(valid_params(), valid_prior_probability())
    def test_prior_probability_property(self, params, prior_probability):
        model = MagicMock(spec=AModel)
        dist = DistributionInMixture(model=model, params=params, prior_probability=prior_probability)

        assert dist.prior_probability == prior_probability

    @given(valid_x(), valid_params())
    def test_pdf_with_none_prior_probability(self, x, params):
        model = MagicMock(spec=AModel)
        dist = DistributionInMixture(model=model, params=params, prior_probability=None)

        result = dist.pdf(x)

        model.params_convert_to_model.assert_not_called()
        model.pdf.assert_not_called()
        assert result == 0.0

    @given(valid_x(), valid_params(), st.floats(min_value=0.01, max_value=1.0))
    def test_pdf_with_float_prior_probability(self, x, params, prior_probability):
        model = MagicMock(spec=AModel)
        converted_params = np.array([3.0, 4.0])
        model.params_convert_to_model.return_value = converted_params
        return_value = 0.5
        model.pdf.return_value = return_value

        dist = DistributionInMixture(model=model, params=params, prior_probability=prior_probability)
        result = dist.pdf(x)

        model.params_convert_to_model.assert_called_once_with(params)
        model.pdf.assert_called_once_with(x, converted_params)
        assert result == pytest.approx(prior_probability * return_value)


class TestIntegrationDistributionInMixture:
    @given(valid_x(), valid_params(), st.floats(min_value=0.01, max_value=1.0))
    def test_pdf_integration(self, x, params, prior_probability):
        model = MockModel()
        dist = DistributionInMixture(model=model, params=params, prior_probability=prior_probability)

        converted_params = model.params_convert_to_model(params)
        expected = prior_probability * model.pdf(x, converted_params)
        actual = dist.pdf(x)

        assert actual == pytest.approx(expected)

    @given(valid_x(), valid_params())
    def test_pdf_with_none_prior_probability_integration(self, x, params):
        model = MockModel()
        dist = DistributionInMixture(model=model, params=params, prior_probability=None)

        actual = dist.pdf(x)

        assert actual == 0.0


class TestMixtureDistribution:
    def test_init(self):
        mock_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        with patch.object(MixtureDistribution, "_normalize") as mock_normalize:
            mixture = MixtureDistribution(distributions=mock_distributions)  # noqa: F841
            mock_normalize.assert_called_once()

    def test_iter(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            iterated_distributions = list(mixture)
            assert iterated_distributions == new_distributions

    def test_getitem(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            for i in range(len(new_distributions)):
                assert mixture[i] == new_distributions[i]

    def test_len(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            assert len(mixture) == len(new_distributions)

    @given(valid_x())
    def test_pdf(self, x):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        pdf_values = [0.1, 0.2, 0.3]

        for dist, val in zip(new_distributions, pdf_values):
            dist.pdf.return_value = val

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            result = mixture.pdf(x)

            for dist in new_distributions:
                dist.pdf.assert_called_once_with(x)

            assert result == pytest.approx(sum(pdf_values))

    def test_has_generator_all_true(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        for dist in new_distributions:
            dist.prior_probability = 0.3
            dist.model = MagicMock(spec=AModelWithGenerator)

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            assert mixture.has_generator is True

    def test_has_generator_one_false(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        for i, dist in enumerate(new_distributions):
            dist.prior_probability = 0.3
            if i == 1:
                dist.model = MagicMock(spec=AModel)
            else:
                dist.model = MagicMock(spec=AModelWithGenerator)

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            assert mixture.has_generator is False

    def test_has_generator_with_none_probability(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        new_distributions[0].prior_probability = None
        new_distributions[0].model = MagicMock(spec=AModel)

        for dist in new_distributions[1:]:
            dist.prior_probability = 0.5
            dist.model = MagicMock(spec=AModelWithGenerator)

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            assert mixture.has_generator is True

    @given(valid_size())
    def test_generate(self, size):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        probabilities = [0.3, 0.5, 0.2]

        for dist, prob in zip(new_distributions, probabilities):
            dist.prior_probability = prob
            dist.has_generator = True
            dist.generate.return_value = np.array([0.1])

        with (
            patch.object(MixtureDistribution, "_normalize"),
            patch("numpy.random.choice") as mock_choice,
            patch("numpy.random.shuffle") as mock_shuffle,
        ):
            mock_choice.return_value = np.array([0, 1, 1, 2])

            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions
            mixture.generate(size=size)

            mock_choice.assert_called_once()
            np.testing.assert_array_equal(mock_choice.call_args[0][0], [0, 1, 2])
            np.testing.assert_array_almost_equal(mock_choice.call_args[1]["p"], probabilities)
            assert mock_choice.call_args[1]["size"] == size

            counts = [1, 2, 1]
            for i, (dist, count) in enumerate(zip(new_distributions, counts)):
                if count > 0:
                    dist.generate.assert_called_once_with(count)

            mock_shuffle.assert_called_once()

    def test_generate_without_generator_raises_typeerror(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(2)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(2)]

        new_distributions[0].prior_probability = 0.5
        new_distributions[0].has_generator = True

        new_distributions[1].prior_probability = 0.5
        new_distributions[1].has_generator = False

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            with pytest.raises(TypeError):
                mixture.generate(size=3)

    def test_normalize_creates_new_distributions(self):
        dists_count = 2
        mock_dist1 = MagicMock(spec=DistributionInMixture)
        mock_dist1.prior_probability = 2.0
        mock_dist1.model = MagicMock(name="model1")
        mock_dist1.params = np.array([1.0])

        mock_dist2 = MagicMock(spec=DistributionInMixture)
        mock_dist2.prior_probability = 3.0
        mock_dist2.model = MagicMock(name="model2")
        mock_dist2.params = np.array([2.0])

        with patch("mpest.core.mixture_distribution.DistributionInMixture") as mock_constructor:
            mixture = MixtureDistribution([mock_dist1, mock_dist2])  # noqa: F841

            assert mock_constructor.call_count == dists_count
            assert mock_constructor.call_args_list[0] == call(mock_dist1.model, mock_dist1.params, 2.0 / 5.0)
            assert mock_constructor.call_args_list[1] == call(mock_dist2.model, mock_dist2.params, 3.0 / 5.0)

    def test_distributions_property(self):
        original_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]
        new_distributions = [MagicMock(spec=DistributionInMixture) for _ in range(3)]

        with patch.object(MixtureDistribution, "_normalize"):
            mixture = MixtureDistribution(distributions=original_distributions)
            mixture._distributions = new_distributions

            assert mixture.distributions == new_distributions


class TestIntegrationMixtureDistribution:
    def test_init_and_normalize(self):
        model = MockModel()
        dist1 = DistributionInMixture(model=model, params=np.array([1.0]), prior_probability=2.0)
        dist2 = DistributionInMixture(model=model, params=np.array([2.0]), prior_probability=3.0)
        dist3 = DistributionInMixture(model=model, params=np.array([3.0]), prior_probability=None)

        mixture = MixtureDistribution([dist1, dist2, dist3])

        assert mixture[0].prior_probability == pytest.approx(2.0 / 5.0)
        assert mixture[1].prior_probability == pytest.approx(3.0 / 5.0)
        assert mixture[2].prior_probability is None

    @given(
        st.lists(valid_params(), min_size=2, max_size=5),
        st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=5),
    )
    def test_from_distributions_integration(self, params_list, probabilities):
        if len(params_list) != len(probabilities):
            probabilities = (
                probabilities[: len(params_list)]
                if len(probabilities) > len(params_list)
                else probabilities + [0.1] * (len(params_list) - len(probabilities))
            )

        model = MockModel()
        distributions = [Distribution(model=model, params=p) for p in params_list]

        mixture = MixtureDistribution.from_distributions(distributions, probabilities)

        assert len(mixture) == len(distributions)

        total_prob = sum(probabilities)
        for i, (dist, prob) in enumerate(zip(mixture, probabilities)):
            assert isinstance(dist, DistributionInMixture)
            assert dist.model is model
            np.testing.assert_array_equal(dist.params, params_list[i])
            assert dist.prior_probability == pytest.approx(prob / total_prob)

    @given(valid_x(), st.integers(min_value=2, max_value=5))
    def test_pdf_integration(self, x, n):
        model = MockModel()
        params = [np.array([float(i)]) for i in range(1, n + 1)]
        priors = [float(i) for i in range(1, n + 1)]

        distributions = [Distribution(model=model, params=p) for p in params]
        mixture = MixtureDistribution.from_distributions(distributions, priors)

        total_prior = sum(priors)
        expected = sum((prior / total_prior) * model.pdf(x, param) for prior, param in zip(priors, params))

        assert mixture.pdf(x) == pytest.approx(expected)

    @given(valid_size())
    def test_generate_integration(self, size):
        model_with_gen = MockModelWithGenerator()

        dist1 = Distribution(model=model_with_gen, params=np.array([1.0]))
        dist2 = Distribution(model=model_with_gen, params=np.array([2.0]))

        mixture = MixtureDistribution.from_distributions([dist1, dist2], [0.3, 0.7])
        result = mixture.generate(size=size)

        assert isinstance(result, np.ndarray)
        assert result.shape == (size,)
        assert result.dtype == np.float64
        assert np.all(result >= 0)
        assert np.all(result < 1)

    def test_generate_without_generator_integration(self):
        model = MockModel()
        model_with_gen = MockModelWithGenerator()

        dist1 = Distribution(model=model, params=np.array([1.0]))
        dist2 = Distribution(model=model_with_gen, params=np.array([2.0]))

        mixture = MixtureDistribution.from_distributions([dist1, dist2], [0.3, 0.7])

        with pytest.raises(TypeError):
            mixture.generate(size=3)
