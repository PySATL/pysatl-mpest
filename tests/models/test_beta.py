import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from mpest.models.beta import Beta


@st.composite
def valid_external_params(draw):
    a = draw(st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False))
    return np.array([a, b])


@st.composite
def valid_internal_params(draw):
    mu = draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
    nu = draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
    return np.array([mu, nu])


def valid_x():
    return st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)


def invalid_x():
    return st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False).filter(
        lambda x: x < 0 or x > 1
    )


class TestBeta:
    @given(valid_external_params())
    def test_params_convert_to_model(self, params):
        beta_model = Beta()
        internal_params = beta_model.params_convert_to_model(params)

        assert np.allclose(internal_params, np.log(params))

    @given(valid_internal_params())
    def test_params_convert_from_model(self, params):
        beta_model = Beta()
        external_params = beta_model.params_convert_from_model(params)

        assert np.allclose(external_params, np.exp(params))

    @given(valid_external_params())
    def test_roundtrip_external_to_internal(self, params):
        beta_model = Beta()
        internal_params = beta_model.params_convert_to_model(params)
        recovered_params = beta_model.params_convert_from_model(internal_params)
        assert np.allclose(recovered_params, params)

    @given(valid_internal_params())
    def test_roundtrip_internal_to_external(self, params):
        beta_model = Beta()
        external_params = beta_model.params_convert_from_model(params)
        recovered_params = beta_model.params_convert_to_model(external_params)
        assert np.allclose(recovered_params, params)

    @given(st.integers(min_value=1000, max_value=10000), valid_external_params())
    def test_generate_range(self, size, params):
        beta_model = Beta()
        samples = beta_model.generate(params, size=size, normalized=False)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    @given(st.integers(min_value=1000, max_value=10000), valid_internal_params())
    def test_generate_normalized_range(self, size, params):
        beta_model = Beta()
        samples = beta_model.generate(params, size=size, normalized=True)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    @pytest.mark.parametrize("mu, nu", [(-0.5, 2.0), (2.0, -0.5), (-1.0, 1.0), (-1.0, -1.0)])
    def test_generate_normalized(self, mu, nu):
        random.seed(42)
        np.random.seed(42)

        beta_model = Beta()
        params = np.array([mu, nu])
        a, b = np.exp(params)
        n_samples = 100000

        samples = beta_model.generate(params, size=n_samples, normalized=True)

        theoretical_dist = stats.beta(a=a, b=b)
        ks_stat, ks_pval = stats.kstest(samples, theoretical_dist.cdf)

        p = 0.01
        assert ks_pval > p

    @pytest.mark.parametrize("a,b", [(0.5, 2.0), (2.0, 0.5), (1.0, 1.0), (5.0, 5.0)])
    def test_generate_not_normalized(self, a, b):
        random.seed(42)
        np.random.seed(42)

        beta_model = Beta()
        params = np.array([a, b])

        n_samples = 100000

        samples = beta_model.generate(params, size=n_samples, normalized=False)

        theoretical_dist = stats.beta(a=a, b=b)
        ks_stat, ks_pval = stats.kstest(samples, theoretical_dist.cdf)

        p = 0.01
        assert ks_pval > p

    @given(st.integers(min_value=1, max_value=100), valid_internal_params())
    def test_generate_shape(self, size, params):
        beta_model = Beta()
        samples = beta_model.generate(params, size=size, normalized=True)
        assert samples.shape == (size,)
        assert samples.dtype == np.float64

    @given(valid_x(), valid_internal_params())
    def test_pdf(self, x, params):
        beta_model = Beta()
        external_params = beta_model.params_convert_from_model(params)
        alpha, beta = external_params

        model_pdf = beta_model.pdf(x, params)
        scipy_pdf = stats.beta.pdf(x, a=alpha, b=beta)

        assert np.isclose(model_pdf, scipy_pdf, rtol=1e-5)

    @given(invalid_x(), valid_internal_params())
    def test_pdf_outside_range(self, x, params):
        beta_model = Beta()
        assert beta_model.pdf(x, params) == 0

    @given(valid_x(), valid_internal_params())
    def test_lpdf(self, x, params):
        beta_model = Beta()
        external_params = beta_model.params_convert_from_model(params)
        alpha, beta = external_params

        model_lpdf = beta_model.lpdf(x, params)
        scipy_lpdf = stats.beta.logpdf(x, a=alpha, b=beta)

        assert np.isclose(model_lpdf, scipy_lpdf, rtol=1e-5)

    @given(invalid_x(), valid_internal_params())
    def test_lpdf_outside_range(self, x, params):
        beta_model = Beta()
        assert beta_model.lpdf(x, params) == -np.inf

    @given(valid_x(), valid_internal_params())
    def test_ldmu(self, x, params):
        beta_model = Beta()

        eps = 1e-6
        params_plus = params.copy()
        params_plus[0] += eps
        params_minus = params.copy()
        params_minus[0] -= eps

        numerical_derivative = (beta_model.lpdf(x, params_plus) - beta_model.lpdf(x, params_minus)) / (2 * eps)
        analytical_derivative = beta_model.ldmu(x, params)

        assert np.isclose(numerical_derivative, analytical_derivative, rtol=1e-3, atol=1e-3)

    @given(valid_x(), valid_internal_params())
    def test_ldnu(self, x, params):
        beta_model = Beta()

        eps = 1e-6
        params_plus = params.copy()
        params_plus[1] += eps
        params_minus = params.copy()
        params_minus[1] -= eps

        numerical_derivative = (beta_model.lpdf(x, params_plus) - beta_model.lpdf(x, params_minus)) / (2 * eps)
        analytical_derivative = beta_model.ldnu(x, params)

        assert np.isclose(numerical_derivative, analytical_derivative, rtol=1e-3, atol=1e-3)

    @given(valid_x(), valid_internal_params())
    def test_ld_params(self, x, params):
        beta_model = Beta()

        derivatives = beta_model.ld_params(x, params)

        assert derivatives.shape == (2,)
        assert derivatives.dtype == np.float64

        assert np.isclose(derivatives[0], beta_model.ldmu(x, params))
        assert np.isclose(derivatives[1], beta_model.ldnu(x, params))

    @given(invalid_x(), valid_internal_params())
    def test_derivatives_outside_range(self, x, params):
        beta_model = Beta()

        assert beta_model.ldmu(x, params) == -np.inf
        assert beta_model.ldnu(x, params) == -np.inf

        derivatives = beta_model.ld_params(x, params)
        assert np.all(derivatives == -np.inf)
