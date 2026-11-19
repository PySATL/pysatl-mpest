"""Unit tests for sampling strategy."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import unittest.mock

import numpy as np
import pytest
from pysatl_core.types import CharacteristicName
from pysatl_mpest.core._sampling import DeterministicSampling


def test_passes_uniform_draws_to_ppf_and_returns_its_result() -> None:
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()
    distr_mock.calculate_characteristic.return_value = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    n = 5
    result = sampler.sample(n=n, distr=distr_mock)

    np.testing.assert_array_equal(result, [2.0, 4.0, 6.0, 8.0, 10.0])
    distr_mock.calculate_characteristic.assert_called_once()

    args, _ = distr_mock.calculate_characteristic.call_args
    assert args[0] == CharacteristicName.PPF

    random_array = args[1]
    assert isinstance(random_array, np.ndarray)
    assert random_array.shape == (n,)
    assert np.all((random_array >= 0.0) & (random_array < 1.0))


def test_integer_seed_reproduces_sample() -> None:
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()
    distr_mock.calculate_characteristic.side_effect = lambda char, arr: arr

    n = 10
    seed = 42

    result1 = sampler.sample(n=n, distr=distr_mock, random_state=seed)
    result2 = sampler.sample(n=n, distr=distr_mock, random_state=seed)

    np.testing.assert_array_equal(result1, result2)


def test_generator_supplies_uniform_draws() -> None:
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()
    distr_mock.calculate_characteristic.side_effect = lambda char, arr: arr

    n = 3
    seed = 123

    rng_provided = np.random.default_rng(seed)
    result = sampler.sample(n=n, distr=distr_mock, random_state=rng_provided)

    rng_manual = np.random.default_rng(seed)
    expected = rng_manual.random(n)

    np.testing.assert_array_equal(result, expected)


def test_default_random_state_produces_distinct_samples() -> None:
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()
    distr_mock.calculate_characteristic.side_effect = lambda char, arr: arr

    n = 5

    result1 = sampler.sample(n=n, distr=distr_mock)
    result2 = sampler.sample(n=n, distr=distr_mock)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(result1, result2)
