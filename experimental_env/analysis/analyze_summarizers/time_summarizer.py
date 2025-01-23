""" A module that provides a class that save data about how the time of estimation is distributed """

from pathlib import Path

import numpy as np
import yaml

from experimental_env.analysis.analyze_summarizers.analysis_summarizer import (
    AnalysisSummarizer,
)
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.utils import round_sig


class TimeSummarizer(AnalysisSummarizer):
    """
    A class that calculates the average error for a dataset using the selected metric.
    """

    def calculate(self, results: list[ExperimentDescription]) -> tuple:
        """
        Helper function for calculating mean and standard deviation of time
        """
        times = []
        for result in results:
            time = np.sum(step.time for step in result.steps)
            times.append(time)

        mean = np.sum(times) / len(times)
        deviation = np.sqrt(np.sum((x - mean) ** 2 for x in times) / len(times))
        return float(mean), float(deviation)

    def analyze_method(self, results: list[ExperimentDescription], method: str):
        mean, deviation = self.calculate(results)

        info_dict = {
            "mean": round_sig(mean, 3),
            "standart_deviation": round_sig(deviation, 3),
        }
        yaml_path: Path = self._out_dir.joinpath("time_info.yaml")

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(info_dict, file)

    def compare_methods(
        self,
        results_1: list[ExperimentDescription],
        results_2: list[ExperimentDescription],
        method_1: str,
        method_2: str,
    ):
        mean_1, deviation_1 = self.calculate(results_1)
        mean_2, deviation_2 = self.calculate(results_2)

        info_dict = {
            f"{method_1}_mean": round_sig(mean_1, 3),
            f"{method_1}_standart_deviation": round_sig(deviation_1, 3),
            f"{method_2}_mean": round_sig(mean_2, 3),
            f"{method_2}_standart_deviation": round_sig(deviation_2, 3),
        }
        yaml_path: Path = self._out_dir.joinpath("time_info.yaml")

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(info_dict, file)
