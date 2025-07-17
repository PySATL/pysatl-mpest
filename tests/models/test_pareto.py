import random

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy import stats

from mpest.models.pareto import Pareto


@st.composite
def valid_external_params(draw):
    x0 = draw(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    k = draw(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False))
    return np.array([x0, k])


@st.composite
def valid_internal_params(draw):
    theta_x0 = draw(st.floats(min_value=-3, max_value=4.5, allow_nan=False, allow_infinity=False))
    theta_k = draw(st.floats(min_value=-3, max_value=2.3, allow_nan=False, allow_infinity=False))
    return np.array([theta_x0, theta_k])


def valid_x():
    return st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False).filter(lambda x: x > 0)


class TestPareto:
    @pytest.fixture
    def pareto_model(self):
        return Pareto()

    def test_get_name(self, pareto_model):
        assert pareto_model.name == "Pareto"

    @given(valid_external_params())
    def test_params_convert_to_model(self, external_params):
        pareto_model = Pareto()

        internal_params = pareto_model.params_convert_to_model(external_params)
        expected = np.log(external_params)
        assert np.allclose(internal_params, expected)

    @given(valid_external_params())
    def test_params_convert_from_model(self, internal_params):
        pareto_model = Pareto()

        external_params = pareto_model.params_convert_from_model(internal_params)
        expected = np.exp(internal_params)
        assert np.allclose(external_params, expected)

    @given(valid_external_params())
    def test_roundtrip_external_to_internal(self, params):
        pareto_model = Pareto()
        internal_params = pareto_model.params_convert_to_model(params)
        recovered_params = pareto_model.params_convert_from_model(internal_params)
        assert np.allclose(recovered_params, params)

    @given(valid_internal_params())
    def test_roundtrip_internal_to_external(self, params):
        pareto_model = Pareto()
        external_params = pareto_model.params_convert_from_model(params)
        recovered_params = pareto_model.params_convert_to_model(external_params)
        assert np.allclose(recovered_params, params)

    @given(valid_internal_params())
    def test_generate_normalized(self, params):
        random.seed(42)
        np.random.seed(42)

        pareto_model = Pareto()

        sample_size = 5000
        samples = pareto_model.generate(params, size=sample_size)

        external_params = pareto_model.params_convert_from_model(params)
        x0, k = external_params

        assert np.all(samples >= x0)

        ks_stat, p_value = stats.kstest(samples, lambda x: stats.pareto.cdf(x, b=k, scale=x0))

        p = 0.01
        assert p_value > p

    @given(valid_external_params())
    def test_generate_not_normalized(self, params):
        random.seed(42)
        np.random.seed(42)

        pareto_model = Pareto()

        sample_size = 5000
        samples = pareto_model.generate(params, size=sample_size, normalized=False)

        x0, k = params

        assert np.all(samples >= x0)

        ks_stat, p_value = stats.kstest(samples, lambda x: stats.pareto.cdf(x, b=k, scale=x0))

        p = 0.01
        assert p_value > p

    @given(st.integers(min_value=1, max_value=100), valid_internal_params())
    def test_generate_shape(self, size, params):
        pareto_model = Pareto()
        samples = pareto_model.generate(params, size=size)
        assert samples.shape == (size,)
        assert samples.dtype == np.float64

    @given(valid_x(), valid_internal_params())
    def test_pdf(self, x, params):
        pareto_model = Pareto()

        external_params = pareto_model.params_convert_from_model(params)
        x0, k = external_params
        assume(x >= x0)

        model_pdf = pareto_model.pdf(x, params)
        scipy_pdf = stats.pareto.pdf(x, b=k, loc=0, scale=x0)

        assert np.isclose(model_pdf, scipy_pdf)

    @given(valid_x(), valid_internal_params())
    def test_pdf_below_x0(self, x, params):
        pareto_model = Pareto()
        external_params = pareto_model.params_convert_from_model(params)
        x0, _ = external_params
        assume(x < x0)

        model_pdf = pareto_model.pdf(x, params)

        assert model_pdf == 0.0

    @given(valid_x(), valid_internal_params())
    def test_lpdf(self, x, params):
        pareto_model = Pareto()
        external_params = pareto_model.params_convert_from_model(params)
        x0, k = external_params
        assume(x >= x0)

        model_lpdf = pareto_model.lpdf(x, params)
        scipy_lpdf = stats.pareto.logpdf(x, b=k, loc=0, scale=x0)

        assert np.isclose(model_lpdf, scipy_lpdf)

    @given(valid_x(), valid_internal_params())
    def test_lpdf_below_x0(self, x, params):
        pareto_model = Pareto()
        external_params = pareto_model.params_convert_from_model(params)
        x0, _ = external_params
        assume(x < x0)

        model_lpdf = pareto_model.lpdf(x, params)

        assert np.isinf(model_lpdf) and model_lpdf < 0

    @given(valid_x(), valid_internal_params())
    def test_derivatives(self, x, params):
        pareto_model = Pareto()
        assume(x > np.exp(params[0]))

        h = 1e-9

        theta_x0_delta_params = params.copy()
        theta_x0_delta_params[0] = params[0] + h

        theta_k_delta_params = params.copy()
        theta_k_delta_params[1] = params[1] + h

        dtheta_x0 = pareto_model.ld_theta_x0(x, params)
        dtheta_k = pareto_model.ld_theta_k(x, params)

        numeric_dtheta_x0 = (pareto_model.lpdf(x, theta_x0_delta_params) - pareto_model.lpdf(x, params)) / h
        numeric_dtheta_k = (pareto_model.lpdf(x, theta_k_delta_params) - pareto_model.lpdf(x, params)) / h

        assert np.isclose(dtheta_x0, numeric_dtheta_x0, 1e-3)
        assert np.isclose(dtheta_k, numeric_dtheta_k, 1e-3)

    @given(valid_x(), valid_internal_params())
    def test_derivatives_below_x0(self, x, params):
        pareto_model = Pareto()
        external_params = pareto_model.params_convert_from_model(params)
        x0, _ = external_params
        assume(x < x0)

        assert np.isinf(pareto_model.ld_theta_x0(x, params))
        assert np.isinf(pareto_model.ld_theta_k(x, params))

    @given(valid_x(), valid_internal_params())
    def test_ld_params(self, x, params):
        pareto_model = Pareto()
        len_grad = 2

        gradient = pareto_model.ld_params(x, params)

        assert len(gradient) == len_grad
        assert np.isclose(gradient[0], pareto_model.ld_theta_x0(x, params))
        assert np.isclose(gradient[1], pareto_model.ld_theta_k(x, params))

    @given(valid_x(), valid_internal_params())
    def test_ld_params_below_x0(self, x, params):
        pareto_model = Pareto()
        external_params = pareto_model.params_convert_from_model(params)
        x0, _ = external_params

        assume(x < x0)

        assert np.isinf(pareto_model.ld_params(x, params)).all()
