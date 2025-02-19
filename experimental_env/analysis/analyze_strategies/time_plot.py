"""A module that provides a class for saving a graph showing the execution time of each step"""

import matplotlib.pyplot as plt

from experimental_env.analysis.analyze_strategies.analysis_strategy import (
    AnalysisStrategy,
)
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.utils import round_sig


class TimePlot(AnalysisStrategy):
    """
    A class that saves a time graph at each step.
    """

    def save_analysis(self):
        plt.legend()
        plt.yscale("log")
        plt.savefig(self._out_dir / "time_plot.png")
        plt.close()

    def overall_analyze_method(self, result: ExperimentDescription, method: str):
        steps = result.steps
        time = [step.time for step in steps]
        plt.plot(time, label=f"Total: {round_sig(sum(time), 2)}s")
        plt.xlabel("Step")
        plt.ylabel("Time [s]")
        self.save_analysis()

    def overall_compare_methods(
        self,
        result_1: ExperimentDescription,
        result_2: ExperimentDescription,
        method_1: str,
        method_2: str,
    ):
        steps_1, steps_2 = result_1.steps, result_2.steps
        time_1 = [step.time for step in steps_1]
        time_2 = [step.time for step in steps_2]

        plt.plot(
            time_1,
            color="green",
            label=f"{method_1}. Total time: {round_sig(sum(time_1), 2)}s",
        )
        plt.plot(
            time_2,
            color="red",
            label=f"{method_2}. Total time: {round_sig(sum(time_2), 2)}s",
        )
        plt.xlabel("Step")
        plt.ylabel("Time [s]")
        self.save_analysis()
