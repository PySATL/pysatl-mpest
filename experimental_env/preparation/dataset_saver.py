"""Class for saving the dataset"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML

from experimental_env.preparation.dataset_description import DatasetDescrciption

matplotlib.use("Agg")


class DatasetSaver:
    """
    A class that saves the dataset along the path selected by the user.
    """

    def __init__(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True)

        self._out_dir = path

    def save_dataset(self, descr: DatasetDescrciption) -> None:
        """
        Save dataset
        """
        samples = descr.samples

        # Save samples
        samples_file: Path = self._out_dir.joinpath("samples.csv")
        np.savetxt(samples_file, samples, delimiter=",")

        # Save config
        yaml = YAML()
        yaml.default_flow_style = False
        config_file: Path = self._out_dir.joinpath("config.yaml")
        with open(config_file, "w", encoding="utf-8") as config:
            yaml.dump(descr.to_yaml_format(), config)

        # Save config
        hist_file = self._out_dir.joinpath("hist.png")
        plt.hist(samples, density=True, color="lightsteelblue")
        plt.savefig(hist_file)
        plt.close()
