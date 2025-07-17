import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy import stats

from mpest.models.uniform import Uniform


@st.composite
def valid_params(draw):
    f1 = draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    f2 = draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))

    tolerance = 1e-5

    assume(abs(f1 - f2) > tolerance)

    a, b = sorted([f1, f2])
    return np.array([a, b])


def valid_x():
    return st.floats(min_value=-150, max_value=150, allow_nan=False, allow_infinity=False)


class TestUniform:
    def test_get_name(self):
        assert Uniform().name == "Uniform"

    @given(valid_params())
    def test_params_convert_to_model(self, params):
        uniform_model = Uniform()
        internal_params = uniform_model.params_convert_to_model(params)
        assert np.allclose(internal_params, params)

    @given(valid_params())
    def test_params_convert_from_model(self, params):
        uniform_model = Uniform()
        external_params = uniform_model.params_convert_from_model(params)
        assert np.allclose(external_params, params)

    @given(valid_params())
    def test_roundtrip_param_conversion(self, params):
        uniform_model = Uniform()

        internal_params = uniform_model.params_convert_to_model(params)
        recovered_params = uniform_model.params_convert_from_model(internal_params)
        assert np.allclose(recovered_params, params)

        external_params = uniform_model.params_convert_from_model(params)
        recovered_internal = uniform_model.params_convert_to_model(external_params)
        assert np.allclose(recovered_internal, params)

    @given(st.integers(min_value=1, max_value=100), valid_params())
    def test_generate_shape_and_type(self, size, params):
        uniform_model = Uniform()

        samples = uniform_model.generate(params, size=size, normalized=True)  # normalized doesn't matter here
        assert samples.shape == (size,)
        assert samples.dtype == np.float64
        samples_non_norm = uniform_model.generate(params, size=size, normalized=False)
        assert samples_non_norm.shape == (size,)
        assert samples_non_norm.dtype == np.float64

    @given(st.integers(min_value=100, max_value=1000), valid_params())
    def test_generate_range(self, size, params):
        uniform_model = Uniform()

        a, b = params
        samples = uniform_model.generate(params, size=size)
        assert np.all(samples >= a)
        assert np.all(samples <= b)

    @given(valid_params())
    def test_generate_distribution(self, params):
        uniform_model = Uniform()

        a, b = params
        n_samples = 5000
        samples = uniform_model.generate(params, size=n_samples)

        theoretical_dist = stats.uniform(loc=a, scale=b - a)
        ks_stat, ks_pval = stats.kstest(samples, theoretical_dist.cdf)

        p_threshold = 0.01
        assert ks_pval > p_threshold

    @given(valid_x(), valid_params())
    def test_pdf(self, x, params):
        uniform_model = Uniform()

        a, b = params
        model_pdf = uniform_model.pdf(x, params)
        if a <= x <= b:
            assume(b > a)
            expected_pdf = 1 / (b - a)
            assert model_pdf == pytest.approx(expected_pdf)
        else:
            expected_pdf = 0
            assert model_pdf == expected_pdf

    @given(valid_x(), valid_params())
    def test_lpdf(self, x, params):
        uniform_model = Uniform()

        a, b = params
        model_lpdf = uniform_model.lpdf(x, params)

        if a <= x <= b:
            assume(b > a)
            expected_lpdf = -np.log(b - a)
            assert model_lpdf == pytest.approx(expected_lpdf)
        else:
            expected_lpdf = -np.inf
            assert model_lpdf == expected_lpdf

    @given(valid_x(), valid_params())
    def test_lda(self, x, params):
        uniform_model = Uniform()
        a, b = params
        model_lda = uniform_model.lda(x, params)

        if a <= x <= b:
            assume(b > a)
            expected_lda = 1.0 / (b - a)
            assert model_lda == pytest.approx(expected_lda)
        else:
            expected_lda = -np.inf
            assert model_lda == expected_lda

    @given(valid_x(), valid_params())
    def test_ldb(self, x, params):
        uniform_model = Uniform()

        a, b = params
        model_ldb = uniform_model.ldb(x, params)

        if a <= x <= b:
            assume(b > a)
            expected_ldb = -1.0 / (b - a)
            assert model_ldb == pytest.approx(expected_ldb)
        else:
            expected_ldb = -np.inf
            assert model_ldb == expected_ldb

    @given(valid_x(), valid_params())
    def test_ld_params(self, x, params):
        uniform_model = Uniform()

        gradient = uniform_model.ld_params(x, params)
        expected_lda = uniform_model.lda(x, params)
        expected_ldb = uniform_model.ldb(x, params)

        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert gradient.dtype == np.float64

        if np.isinf(expected_lda):
            assert gradient[0] == -np.inf
        else:
            assert gradient[0] == pytest.approx(expected_lda)

        if np.isinf(expected_ldb):
            assert gradient[1] == -np.inf
        else:
            assert gradient[1] == pytest.approx(expected_ldb)

        assert np.allclose(gradient, np.array([expected_lda, expected_ldb]), equal_nan=False)
