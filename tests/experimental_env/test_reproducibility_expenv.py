import tempfile
from pathlib import Path

import numpy as np

from experimental_env.experiment.estimators import LikelihoodEstimator, LMomentsEstimator
from experimental_env.experiment.experiment_description import StepDescription
from experimental_env.experiment.experiment_executors.random_executor import RandomExperimentExecutor
from experimental_env.experiment.experiment_parser import ExperimentParser
from experimental_env.preparation.dataset_generator import RandomDatasetGenerator
from experimental_env.preparation.dataset_parser import SamplesDatasetParser
from mpest import MixtureDistribution
from mpest.em.breakpointers import StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker, PriorProbabilityThresholdChecker
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp


def compare_mixtures(mxt_1: MixtureDistribution, mxt_2: MixtureDistribution):
    return all(
        (np.array_equal(d1.prior_probability, d2.prior_probability, equal_nan=True))
        and np.array_equal(d1.params, d2.params, equal_nan=True)
        and (type(d1.model) is type(d2.model))
        for d1, d2 in zip(mxt_1, mxt_2)
    )


def compare_mixtures_without_priors(mxt_1: MixtureDistribution, mxt_2: MixtureDistribution):
    return all(
        np.array_equal(d1.params, d2.params, equal_nan=True) and (type(d1.model) is type(d2.model))
        for d1, d2 in zip(mxt_1, mxt_2)
    )


def compare_steps(step_descr_1: list[StepDescription], step_descr_2: list[StepDescription]):
    return all(
        compare_mixtures(step_1.result_mixture, step_2.result_mixture)
        for step_1, step_2 in zip(step_descr_1, step_descr_2)
    )


def stage_1(working_dir):
    WORKING_DIR = Path(working_dir) / "stage_1"
    SAMPLES_SIZE = 200

    r_generator = RandomDatasetGenerator(42)
    mixtures = [
        [GaussianModel, WeibullModelExp],
        [ExponentialModel, GaussianModel],
        [ExponentialModel, ExponentialModel],
    ]
    for models in mixtures:
        r_generator.generate(SAMPLES_SIZE, models, WORKING_DIR, exp_count=5)


def stage_2(working_dir):
    WORKING_DIR = Path(working_dir) / "stage_2"
    SOURCE_DIR = Path(working_dir) / "stage_1"
    parser = SamplesDatasetParser()
    datasets = parser.parse(SOURCE_DIR)

    executor = RandomExperimentExecutor(WORKING_DIR, 5, 43)
    executor.execute(
        datasets,
        LikelihoodEstimator(
            StepCountBreakpointer(max_step=16),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
        ),
    )

    executor = RandomExperimentExecutor(WORKING_DIR, 5, 43)
    executor.execute(
        datasets,
        LMomentsEstimator(
            StepCountBreakpointer(max_step=16),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
        ),
    )


def test_expenv_scenario():
    ds_1 = None
    ds_2 = None

    ELM_exp_1 = None
    EM_exp_1 = None
    ELM_exp_2 = None
    EM_exp_2 = None

    with tempfile.TemporaryDirectory() as tmpdir:
        stage_1(tmpdir)
        stage_1_dir = Path(tmpdir) / "stage_1"
        parser = SamplesDatasetParser()
        ds_1 = parser.parse(stage_1_dir)

        stage_2(tmpdir)
        ELM_stage_2_dir = Path(tmpdir) / "stage_2" / "ELM"
        EM_stage_2_dir = Path(tmpdir) / "stage_2" / "MLE-EM"
        ELM_exp_1 = ExperimentParser().parse(ELM_stage_2_dir)
        EM_exp_1 = ExperimentParser().parse(EM_stage_2_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        stage_1(tmpdir)
        stage_1_dir = Path(tmpdir) / "stage_1"
        parser = SamplesDatasetParser()
        ds_2 = parser.parse(stage_1_dir)

        stage_2(tmpdir)
        ELM_stage_2_dir = Path(tmpdir) / "stage_2" / "ELM"
        EM_stage_2_dir = Path(tmpdir) / "stage_2" / "MLE-EM"
        ELM_exp_2 = ExperimentParser().parse(ELM_stage_2_dir)
        EM_exp_2 = ExperimentParser().parse(EM_stage_2_dir)

    for mxt in ds_1:
        res_1, res_2 = ds_1[mxt], ds_2[mxt]
        for descr_1, descr_2 in zip(res_1, res_2):
            assert np.array_equal(descr_1.samples, descr_2.samples)
            assert descr_1.exp_num == descr_2.exp_num
            assert compare_mixtures(descr_1.base_mixture, descr_2.base_mixture)

    for mxt in ELM_exp_1:
        res_1, res_2 = ELM_exp_1[mxt], EM_exp_1[mxt]
        for exp_1, exp_2 in zip(res_1, res_2):
            assert compare_mixtures(exp_1.base_mixture, exp_2.base_mixture)
            assert compare_mixtures(exp_1.init_mixture, exp_2.init_mixture)
            assert not compare_mixtures_without_priors(exp_1.base_mixture, exp_1.init_mixture)
            assert not compare_mixtures_without_priors(exp_2.base_mixture, exp_2.init_mixture)
            assert exp_1.samples_size == exp_2.samples_size
            assert exp_1.exp_num == exp_2.exp_num

    for mxt in ELM_exp_1:
        res_1, res_2 = ELM_exp_1[mxt], ELM_exp_2[mxt]
        for exp_1, exp_2 in zip(res_1, res_2):
            assert compare_mixtures(exp_1.base_mixture, exp_2.base_mixture)
            assert compare_mixtures(exp_1.init_mixture, exp_2.init_mixture)
            assert compare_steps(exp_1.steps, exp_2.steps)
            assert exp_1.samples_size == exp_2.samples_size
            assert exp_1.exp_num == exp_2.exp_num

    for mxt in ELM_exp_1:
        res_1, res_2 = EM_exp_1[mxt], EM_exp_2[mxt]
        for exp_1, exp_2 in zip(res_1, res_2):
            assert compare_mixtures(exp_1.base_mixture, exp_2.base_mixture)
            assert compare_mixtures(exp_1.init_mixture, exp_2.init_mixture)
            assert compare_steps(exp_1.steps, exp_2.steps)
            assert exp_1.samples_size == exp_2.samples_size
            assert exp_1.exp_num == exp_2.exp_num
