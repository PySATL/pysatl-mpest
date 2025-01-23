""" A module that provides a class that save data about how the metric error is distributed """

from math import isnan
from pathlib import Path

import numpy as np
import yaml

from experimental_env.analysis.analyze_summarizers.analysis_summarizer import (
    AnalysisSummarizer,
)
from experimental_env.analysis.metrics import AMetric
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.utils import round_sig


class ErrorSummarizer(AnalysisSummarizer):
    """
    A class that calculates the average error for a dataset using the selected metric.
    """

    def __init__(self, metric: AMetric):
        super().__init__()
        self._metric = metric

    def calculate(self, results: list[ExperimentDescription]) -> tuple:
        """
        Helper function for calculating mean and standard deviation
        """
        errors = []
        for result in results:
            base_mixture = result.base_mixture
            result_mixture = result.steps[-1].result_mixture
            error = self._metric.error(base_mixture, result_mixture)

            if isnan(error):
                continue

            errors.append(error)

        mean = np.sum(errors) / len(errors)
        standart_deviation = np.sqrt(
            np.sum((x - mean) ** 2 for x in errors) / len(errors)
        )

        errors.sort()
        median = errors[len(errors) // 2]

        return float(mean), float(standart_deviation), float(median)

    def analyze_method(self, results: list[ExperimentDescription], method: str):
        mean, deviation, median = self.calculate(results)

        info_dict = {
            "mean": round_sig(mean, 3),
            "standart_deviation": round_sig(deviation, 3),
            "median": round_sig(median, 3),
        }
        yaml_path: Path = self._out_dir.joinpath("metric_info.yaml")

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(info_dict, file)

    def compare_methods(
        self,
        results_1: list[ExperimentDescription],
        results_2: list[ExperimentDescription],
        method_1: str,
        method_2: str,
    ):
        mean_1, deviation_1, median_1 = self.calculate(results_1)
        mean_2, deviation_2, median_2 = self.calculate(results_2)

        info_dict = {
            f"{method_1}_mean": round_sig(mean_1, 3),
            f"{method_1}_standart_deviation": round_sig(deviation_1, 3),
            f"{method_1}_median": round_sig(median_1, 3),
            f"{method_2}_mean": round_sig(mean_2, 3),
            f"{method_2}_standart_deviation": round_sig(deviation_2, 3),
            f"{method_2}_median": round_sig(median_2, 3),
        }
        yaml_path: Path = self._out_dir.joinpath("metric_info.yaml")

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(info_dict, file)
