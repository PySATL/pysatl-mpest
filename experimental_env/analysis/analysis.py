"""A module that provides a class for performing the third stage of the experiment"""

from pathlib import Path

from tqdm import tqdm

from experimental_env.analysis.analyze_strategies.analysis_strategy import (
    AnalysisStrategy,
)
from experimental_env.analysis.analyze_summarizers.analysis_summarizer import (
    AnalysisSummarizer,
)
from experimental_env.experiment.experiment_description import ExperimentDescription

ParserOutput = dict[str, list[ExperimentDescription]]


class Analysis:
    """
    A class that provides a method for analyzing a method or for comparing two methods
    """

    def __init__(
        self,
        path: Path,
        actions: list[AnalysisStrategy],
        summarizers: list[AnalysisSummarizer],
    ):
        self._out_dir = path
        self._actions = actions
        self._summarizers = summarizers

    def analyze(self, results: ParserOutput, method: str):
        """
        A function for analyzing the method

        :param results: The result of the method on the second stage of the experiment,
         which was obtained using a parser.
        :param method: Name of analyzed method
        """
        method_dir: Path = self._out_dir.joinpath(method)

        for mixture_name, exp_descriptions in results.items():
            mixture_dir: Path = method_dir.joinpath(mixture_name)

            for summarizer in self._summarizers:
                summarizer.set_path(mixture_dir)
                summarizer.analyze_method(exp_descriptions, method)

            with tqdm(total=len(exp_descriptions)) as pbar:
                for exp_descr in exp_descriptions:
                    pbar.update()
                    exp_dir: Path = mixture_dir.joinpath(f"experiment_{exp_descr.exp_num}")

                    for action in self._actions:
                        action.set_path(exp_dir)
                        action.overall_analyze_method(exp_descr, method)

    def compare(
        self,
        results_1: ParserOutput,
        results_2: ParserOutput,
        method_1: str,
        method_2: str,
    ):
        """
        A function for comparing the methods

        :param results_1: The result of the first method on the second stage of the experiment,
         which was obtained using a parser.
        :param results_2: The result of the second method on the second stage of the experiment,
         which was obtained using a parser.
        :param method_1: Name of the first method
        :param method_2: Name of the second method
        """
        method_dir = self._out_dir.joinpath(f"{method_1} + {method_2}")

        for mixture_name in results_1:
            mixture_dir = method_dir.joinpath(mixture_name)

            for summarizer in self._summarizers:
                summarizer.set_path(mixture_dir)
                summarizer.compare_methods(results_1[mixture_name], results_2[mixture_name], method_1, method_2)

            with tqdm(total=len(results_1[mixture_name])) as pbar:
                for res in zip(results_1[mixture_name], results_2[mixture_name]):
                    pbar.update()
                    exp_dir = mixture_dir.joinpath(f"experiment_{res[0].exp_num}")

                    for action in self._actions:
                        action.set_path(exp_dir)
                        action.overall_compare_methods(res[0], res[1], method_1, method_2)
