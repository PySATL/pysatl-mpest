import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from mpest.models.cauchy import Cauchy
from scipy import stats
from scipy.stats import cauchy


@st.composite
def valid_external_params(draw):
    x0 = draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    gamma = draw(st.floats(min_value=1e-5, max_value=100, allow_nan=False, allow_infinity=False))
    return np.array([x0, gamma])


@st.composite
def valid_internal_params(draw):
    x0 = draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    sigma = draw(st.floats(min_value=-10, max_value=5, allow_nan=False, allow_infinity=False))
    return np.array([x0, sigma])


def valid_x():
    return st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)


class TestCauchy:
    def test_get_name(self):
        cauchy_model = Cauchy()
        assert cauchy_model.name == "Cauchy"

    @given(valid_external_params())
    def test_params_convert_to_model(self, params):
        cauchy_model = Cauchy()

        internal_params = cauchy_model.params_convert_to_model(params)
        expected = params[0], np.log(params[1])
        assert np.allclose(internal_params, expected)

    @given(valid_internal_params())
    def test_params_convert_from_model(self, params):
        cauchy_model = Cauchy()

        external_params = cauchy_model.params_convert_from_model(params)
        expected = params[0], np.exp(params[1])
        assert np.allclose(external_params, expected)

    @given(valid_external_params())
    def test_roundtrip_external_to_internal(self, params):
        cauchy_model = Cauchy()

        internal_params = cauchy_model.params_convert_to_model(params)
        recovered_params = cauchy_model.params_convert_from_model(internal_params)
        assert np.allclose(recovered_params, params)

    @given(valid_internal_params())
    def test_roundtrip_internal_to_external(self, params):
        cauchy_model = Cauchy()

        external_params = cauchy_model.params_convert_from_model(params)
        recovered_params = cauchy_model.params_convert_to_model(external_params)
        assert np.allclose(recovered_params, params)

    @given(valid_internal_params())
    def test_generate_normalized(self, params):
        random.seed(42)
        np.random.seed(42)

        cauchy_model = Cauchy()
        n_samples = 10000
        samples = cauchy_model.generate(params, size=n_samples)

        x0, gamma = cauchy_model.params_convert_from_model(params)

        left_tail = samples[samples < x0]
        right_tail = 2 * x0 - samples[samples > x0]

        ks_stat, ks_pval = stats.kstest(left_tail, right_tail)
        p = 0.01
        assert ks_pval > p

        theoretical_dist = stats.cauchy(loc=x0, scale=gamma)
        ks_stat, ks_pval = stats.kstest(samples, theoretical_dist.cdf)
        assert ks_pval > p

    @given(valid_external_params())
    def test_generate_not_normalized(self, params):
        random.seed(42)
        np.random.seed(42)

        x0, gamma = params
        cauchy_model = Cauchy()
        n_samples = 10000
        samples = cauchy_model.generate(params, size=n_samples, normalized=False)

        left_tail = samples[samples < x0]
        right_tail = 2 * x0 - samples[samples > x0]

        ks_stat, ks_pval = stats.kstest(left_tail, right_tail)
        p = 0.01
        assert ks_pval > p

        theoretical_dist = stats.cauchy(loc=x0, scale=gamma)
        ks_stat, ks_pval = stats.kstest(samples, theoretical_dist.cdf)
        assert ks_pval > p

    @given(st.integers(min_value=1, max_value=100), valid_internal_params())
    def test_generate_shape(self, size, params):
        cauchy_model = Cauchy()
        samples = cauchy_model.generate(params, size=size, normalized=True)
        assert samples.shape == (size,)
        assert samples.dtype == np.float64

    @given(valid_x(), valid_internal_params())
    def test_pdf(self, x, params):
        cauchy_model = Cauchy()
        external_params = cauchy_model.params_convert_from_model(params)
        x0, gamma = external_params

        model_pdf = cauchy_model.pdf(x, params)
        scipy_pdf = stats.cauchy.pdf(x, loc=x0, scale=gamma)
        assert model_pdf == pytest.approx(scipy_pdf)

    @given(valid_x(), valid_internal_params())
    def test_pdf_positive(self, x, params):
        cauchy_model = Cauchy()
        result = cauchy_model.pdf(x, params)
        assert result > 0

    @given(valid_x(), valid_internal_params())
    def test_pdf_symmetry(self, x, params):
        cauchy_model = Cauchy()
        x0 = params[0]
        left = cauchy_model.pdf(x0 - x, params)
        right = cauchy_model.pdf(x0 + x, params)
        assert left == pytest.approx(right)

    @given(valid_x(), valid_internal_params())
    def test_lpdf(self, x, params):
        cauchy_model = Cauchy()
        external_params = cauchy_model.params_convert_from_model(params)
        x0, gamma = external_params

        model_lpdf = cauchy_model.lpdf(x, params)
        scipy_lpdf = stats.cauchy.logpdf(x, loc=x0, scale=gamma)

        assert model_lpdf == pytest.approx(scipy_lpdf)

    @given(valid_internal_params())
    def test_ldx0_at_x0(self, params):
        cauchy_model = Cauchy()
        x0 = params[0]
        result = cauchy_model.ldx0(x0, params)
        assert result == pytest.approx(0.0)

    @given(valid_x(), valid_internal_params())
    def test_derivatives(self, x, params):
        cauchy_model = Cauchy()
        external_params = cauchy_model.params_convert_from_model(params)
        x0, gamma = external_params

        dx0 = cauchy_model.ldx0(x, params)
        dsigma = cauchy_model.ldsigma(x, params)

        h = 1e-9
        numeric_dx0 = (cauchy.logpdf(x, loc=x0 + h, scale=gamma) - cauchy.logpdf(x, loc=x0 - h, scale=gamma)) / (2 * h)

        numeric_dgamma = (cauchy.logpdf(x, loc=x0, scale=gamma + h) - cauchy.logpdf(x, loc=x0, scale=gamma - h)) / (
            2 * h
        )
        numeric_dsigma = numeric_dgamma * gamma

        assert dx0 == pytest.approx(numeric_dx0, abs=1e-3)
        assert dsigma == pytest.approx(numeric_dsigma, abs=1e-3)

    @given(valid_x(), valid_internal_params())
    def test_ld_params(self, x, params):
        cauchy_model = Cauchy()
        gradient = cauchy_model.ld_params(x, params)
        params_number = 2
        assert len(gradient) == params_number
        assert gradient[0] == pytest.approx(cauchy_model.ldx0(x, params))
        assert gradient[1] == pytest.approx(cauchy_model.ldsigma(x, params))
