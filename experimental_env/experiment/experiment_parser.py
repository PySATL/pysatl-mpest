"""A module containing a class for parsing the second stage of the experiment"""

import os
from pathlib import Path

import yaml
from numpy import genfromtxt

from experimental_env.analysis.analysis import ParserOutput
from experimental_env.experiment.experiment_description import (
    ExperimentDescription,
    StepDescription,
)
from experimental_env.preparation.dataset_description import DatasetDescrciption
from experimental_env.utils import create_mixture_by_key, sort_human


class ExperimentParser:
    """
    A class that parses the second stage of the experiment.
    """

    def _get_steps(self, exp_dir: Path):
        steps = []
        for step in sort_human(os.listdir(exp_dir)):
            if "step" not in step:
                continue
            step_dir = exp_dir.joinpath(step)
            step_config_p = step_dir.joinpath("config.yaml")

            # Get step
            with open(step_config_p, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
                step_time = config["time"]
                step_mixture = create_mixture_by_key(config, "step_distributions")

            steps.append(StepDescription(step_mixture, step_time))

        return steps

    def parse(self, path: Path) -> ParserOutput:
        """
        The parsing method

        :param path: The path to the directory of the second stage of the experiment
        """
        output = {}

        # Open each mixture dir
        for mixture_name in os.listdir(path):
            mixture_name_dir: Path = path.joinpath(mixture_name)
            mixture_name_list = []

            # Open each experiment dir
            for exp_name in os.listdir(mixture_name_dir):
                experiment_dir: Path = mixture_name_dir.joinpath(exp_name)
                samples_p: Path = experiment_dir.joinpath("samples.csv")
                config_p: Path = experiment_dir.joinpath("config.yaml")

                # Get samples
                samples = genfromtxt(samples_p, delimiter=",")

                # Get base mixture
                with open(config_p, encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)
                    samples_size = config["samples_size"]
                    exp_num = config["exp_num"]
                    error = config["error"]

                    base_mixture = create_mixture_by_key(config, "distributions")
                    init_mixture = create_mixture_by_key(config, "init_distributions")

                # Get all steps
                steps = self._get_steps(experiment_dir)

                # Save results of experiment
                ds_descr = DatasetDescrciption(samples_size, samples, base_mixture, exp_num)

                result_descr = ExperimentDescription.from_steps(init_mixture, steps, ds_descr, error)

                mixture_name_list.append(result_descr)

            output[mixture_name] = mixture_name_list
        return output
