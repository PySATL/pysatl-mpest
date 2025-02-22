"""A module with saver for second stage of experiment"""

from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from experimental_env.experiment.experiment_description import ExperimentDescription


class ExperimentSaver:
    """
    A class that implements saving the results of the second stage
    """

    def __init__(self, path: Path):
        self._out_dir = path

        if not path.exists():
            path.mkdir(parents=True)

    def save(self, descr: ExperimentDescription):
        """
        A function that implements saving.

        :param descr: An object containing information about the current experiment
        """
        yaml = YAML()
        yaml.default_flow_style = False

        with open(self._out_dir / "config.yaml", "w", encoding="utf-8") as config:
            yaml.dump(descr.to_yaml_format(), config)

        np.savetxt(self._out_dir / "samples.csv", descr.samples, delimiter=",")

        for i, step_descr in enumerate(descr):
            step_dir: Path = self._out_dir.joinpath(f"step_{i + 1}")
            if not step_dir.exists():
                step_dir.mkdir()

            # Save step config
            yaml_path: Path = step_dir.joinpath("config.yaml")

            with open(yaml_path, "w", encoding="utf-8") as config:
                yaml.dump(step_descr.to_yaml_format(), config)
