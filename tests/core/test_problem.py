from unittest.mock import Mock

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from mpest.core.problem import Problem


@st.composite
def valid_samples(draw, min_size=1, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    samples_list = draw(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )

    return np.array(samples_list)


@st.composite
def valid_distributions(draw, min_count=1, max_count=5):
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    distributions = []
    for _ in range(count):
        mock_dist = Mock()
        mock_dist.name = draw(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll")))
        )
        distributions.append(mock_dist)
    return distributions


class TestProblem:
    def test_get_instance(self):
        samples = np.array([1.0, 2.0, 3.0])
        distributions = [Mock(), Mock()]

        problem = Problem(samples=samples, distributions=distributions)

        assert isinstance(problem, Problem)

    @given(valid_samples(), valid_distributions())
    def test_init(self, samples, distributions):
        """Тест инициализации Problem."""
        problem = Problem(samples=samples, distributions=distributions)

        assert problem.samples is samples
        assert problem.distributions is distributions

    @given(valid_samples(), valid_distributions())
    def test_samples_property(self, samples, distributions):
        problem = Problem(samples=samples, distributions=distributions)

        assert problem.samples is samples
        assert np.array_equal(problem.samples, samples)

    @given(valid_samples(), valid_distributions())
    def test_distributions_property(self, samples, distributions):
        problem = Problem(samples=samples, distributions=distributions)

        assert problem.distributions is distributions
        assert len(problem.distributions) == len(distributions)
        for i, dist in enumerate(distributions):
            assert problem.distributions[i] is dist
