"""A module providing a class for saving the convergence graph of the error function."""

from matplotlib import pyplot as plt

from experimental_env.analysis.analyze_strategies.analysis_strategy import (
    AnalysisStrategy,
)
from experimental_env.analysis.metrics import AMetric
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.utils import round_sig


class ErrorConvergence(AnalysisStrategy):
    """
    A class that saves graphs with errors at each step.
    The constructor accepts an error function that satisfies the AMetric interface.
    """

    def __init__(self, metric: AMetric):
        super().__init__()
        self._metric = metric

    def calculate_errors(self, result: ExperimentDescription) -> list[float]:
        """
        Helper function for calculating metric errors
        """
        steps = result.steps
        base_mixture = result.base_mixture
        mixtures = [step.result_mixture for step in steps]

        return [self._metric.error(base_mixture, mxt) for mxt in mixtures]

    def save_analysis(self):
        plt.legend()
        plt.yscale("log")
        plt.savefig(self._out_dir / "error_convergence_plot.png")
        plt.close()

    def overall_analyze_method(self, result: ExperimentDescription, method: str):
        plt.title(self._metric.name)
        plt.xlabel("Step")
        plt.ylabel("Error")

        metric_errors = self.calculate_errors(result)
        average = sum(metric_errors) / len(metric_errors)

        plt.plot(metric_errors, label=f"Average: {round_sig(average, 2)}")

        self.save_analysis()

    def overall_compare_methods(
        self,
        result_1: ExperimentDescription,
        result_2: ExperimentDescription,
        method_1: str,
        method_2: str,
    ):
        metric_errors_1 = self.calculate_errors(result_1)
        metric_errors_2 = self.calculate_errors(result_2)

        plt.title(self._metric.name)
        plt.xlabel("Step")
        plt.ylabel("Error")

        average_1 = sum(metric_errors_1) / len(metric_errors_1)
        average_2 = sum(metric_errors_2) / len(metric_errors_2)

        plt.plot(
            metric_errors_1,
            label=f"{method_1}. Average: {round_sig(average_1, 2)}",
        )
        plt.plot(
            metric_errors_2,
            label=f"{method_2}. Average: {round_sig(average_2, 2)}",
        )

        self.save_analysis()
