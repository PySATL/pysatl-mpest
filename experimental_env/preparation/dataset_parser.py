""" A module containing a class for parsing datasets from the first stage of the experiment """
import os
from pathlib import Path

import yaml
from numpy import genfromtxt

from experimental_env.preparation.dataset_description import DatasetDescrciption
from experimental_env.utils import create_mixture_by_key


class SamplesDatasetParser:
    """
    A class that provides a method for parsing.
    """

    def parse(self, path: Path) -> dict:
        """
        A function that implements parsing.

        :param path: Path to datasets
        """
        # pylint: disable = duplicate-code
        output = {}
        # Open each mixture dir
        for mixture_name in os.listdir(path):
            mixture_name_dir: Path = path.joinpath(mixture_name)
            mixture_name_dict = []

            # Open each experiment dir
            for exp in os.listdir(mixture_name_dir):
                # TODO: For macOS, there is a problem of finding files.DS_Store at which the program crashes. It is necessary to sort such files.
                experiment_dir: Path = mixture_name_dir.joinpath(exp)
                samples_p: Path = experiment_dir.joinpath("samples.csv")
                config_p: Path = experiment_dir.joinpath("config.yaml")

                # Get samples
                samples = genfromtxt(samples_p, delimiter=",")

                # Get mixture
                with open(config_p, "r", encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)
                    exp_count = config["exp_num"]
                    samples_size = config["samples_size"]
                    base_mixture = create_mixture_by_key(config, "distributions")

                mixture_name_dict.append(
                    DatasetDescrciption(samples_size, samples, base_mixture, exp_count)
                )

            output[mixture_name] = mixture_name_dict
        return output
