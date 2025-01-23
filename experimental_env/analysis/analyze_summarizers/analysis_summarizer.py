""" A module containing an abstract summarizer class for analysis methods. """

from abc import ABC, abstractmethod
from pathlib import Path

from experimental_env.experiment.experiment_description import ExperimentDescription


class AnalysisSummarizer(ABC):
    """
    A class that provides functionality for analyzing methods by examining the overall set of experimental results for a given mixture.
    """

    def __init__(self):
        self._out_dir = "/"

    def set_path(self, path: Path):
        """
        Set the directory where the results of the analysis or comparison will be saved

        :param path: The path to the directory
        """

        if not path.exists():
            path.mkdir(parents=True)

        self._out_dir = path

    @abstractmethod
    def analyze_method(self, results: list[ExperimentDescription], method: str):
        """
        Analysis of the method for all experiments on this mixture.

        :param results: The results of the method on the second stage of the experiment, which was obtained using a parser.
        :param method: The name of the method that we are analyzing
        """

        raise NotImplementedError

    @abstractmethod
    def compare_methods(
        self,
        results_1: list[ExperimentDescription],
        results_2: list[ExperimentDescription],
        method_1: str,
        method_2: str,
    ):
        """
        A function for comparing the methods

        :param results_1: The results of the first method on the second stage of the experiment, which was obtained using a parser.
        :param results_2: The results of the second method on the second stage of the experiment, which was obtained using a parser.
        :param method_1: Name of the first method
        :param method_2: Name of the second method
        """

        raise NotImplementedError
