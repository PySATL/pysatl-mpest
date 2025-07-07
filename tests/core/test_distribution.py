from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mpest.core.distribution import Distribution
from mpest.models import AModel, AModelWithGenerator


@st.composite
def valid_params(draw, min_size=1, max_size=5):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    params_list = draw(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(params_list)


@st.composite
def valid_x(draw):
    return draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))


@st.composite
def valid_size(draw):
    return draw(st.integers(min_value=1, max_value=100))


class MockModel(AModel):
    @property
    def name(self):
        return "MockModel"

    def pdf(self, x, params):
        return 0.1 * x * sum(params)

    def lpdf(self, x, params):
        return np.log(self.pdf(x, params))

    def params_convert_to_model(self, params):
        return params

    def params_convert_from_model(self, params):
        return params


class MockModelWithGenerator(AModelWithGenerator):
    @property
    def name(self):
        return "MockModelWithGenerator"

    def pdf(self, x, params):
        return 0.1 * x * sum(params)

    def lpdf(self, x, params):
        return np.log(self.pdf(x, params))

    def params_convert_to_model(self, params):
        return params

    def params_convert_from_model(self, params):
        return params

    def generate(self, params, size=1, **kwargs):
        return np.random.uniform(0, 1, size=size)


class TestModuleDistribution:
    def test_init(self):
        model = Mock()
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        assert dist._model is model
        assert np.array_equal(dist._params, params)

    def test_from_params(self):
        MockModelClass = Mock()
        mock_instance = Mock()
        MockModelClass.return_value = mock_instance
        params = [1.0, 2.0]

        dist = Distribution.from_params(MockModelClass, params)

        MockModelClass.assert_called_once()
        assert dist._model is mock_instance
        assert np.array_equal(dist._params, np.array(params))

    def test_model_property(self):
        model = Mock()
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        assert dist.model is model

    def test_params_property(self):
        model = Mock()
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        assert dist.params is params
        assert np.array_equal(dist.params, params)

    def test_has_generator_property_true(self):
        model = MagicMock(spec=AModelWithGenerator)
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        assert dist.has_generator is True

    def test_has_generator_property_false(self):
        model = MagicMock(spec=AModel)
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        assert dist.has_generator is False

    @given(valid_x(), valid_params())
    def test_pdf_calls_model_pdf_correctly(self, x, params):
        model = Mock()
        return_value = 0.1
        converted_params = np.array([3.0, 4.0])
        model.params_convert_to_model.return_value = converted_params
        model.pdf.return_value = return_value

        dist = Distribution(model=model, params=params)
        result = dist.pdf(x)

        model.params_convert_to_model.assert_called_once_with(params)
        model.pdf.assert_called_once_with(x, converted_params)
        assert result == return_value

    @given(valid_size(), valid_params())
    def test_generate_with_generator_model(self, size, params):
        model = MagicMock(spec=AModelWithGenerator)
        converted_params = np.array([3.0, 4.0])
        model.params_convert_to_model.return_value = converted_params
        generated_samples = np.random.uniform(0, 1, size=size)
        model.generate.return_value = generated_samples

        dist = Distribution(model=model, params=params)
        result = dist.generate(size=size)

        model.params_convert_to_model.assert_called_once_with(params)
        model.generate.assert_called_once_with(converted_params, size=size)
        assert np.array_equal(result, generated_samples)

    def test_generate_without_generator_raises_typeerror(self):
        model = MagicMock(spec=AModel)
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        with pytest.raises(TypeError):
            dist.generate(size=3)


class TestIntegrationDistribution:
    @given(valid_x(), valid_params())
    def test_pdf_integration(self, x, params):
        model = MockModel()
        dist = Distribution(model=model, params=params)

        converted_params = model.params_convert_to_model(params)
        expected = model.pdf(x, converted_params)
        actual = dist.pdf(x)

        assert actual == pytest.approx(expected)

    @given(valid_size(), valid_params())
    def test_generate_integration(self, size, params):
        model = MockModelWithGenerator()

        dist = Distribution(model=model, params=params)
        result = dist.generate(size=size)

        assert result.shape == (size,)
        assert result.dtype == np.float64
        assert np.all(result >= 0)
        assert np.all(result < 1)

    def test_generate_without_generator_raises_typeerror_integration(self):
        model = MockModel()
        params = np.array([1.0, 2.0])

        dist = Distribution(model=model, params=params)

        with pytest.raises(TypeError):
            dist.generate(size=3)

    @given(valid_x(), valid_params())
    def test_pdf_consistent_results(self, x, params):
        model = MockModel()
        dist = Distribution(model=model, params=params)

        result1 = dist.pdf(x)
        result2 = dist.pdf(x)

        assert result1 == pytest.approx(result2)
