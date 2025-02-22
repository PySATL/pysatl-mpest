"""A module providing a class for saving a distribution density graph."""

import matplotlib.pyplot as plt
import numpy as np

from experimental_env.analysis.analyze_strategies.analysis_strategy import (
    AnalysisStrategy,
)
from experimental_env.experiment.experiment_description import ExperimentDescription


class DensityPlot(AnalysisStrategy):
    """
    A class providing a methods for saving a distribution density graph.
    """

    def __init__(self, y_scale="linear"):
        super().__init__()
        self._y_scale = y_scale

    def save_analysis(self):
        plt.legend()
        plt.yscale(self._y_scale)
        plt.savefig(self._out_dir / "density_plot.png")
        plt.close()

    def overall_analyze_method(self, result: ExperimentDescription, method: str):
        x_linspace = np.linspace(-10, 10, 1000)

        plt.hist(result.samples, color="lightsteelblue", density=True)
        plt.plot(
            x_linspace,
            [result.base_mixture.pdf(x) for x in x_linspace],
            color="red",
            label="base",
        )
        plt.plot(
            x_linspace,
            [result.steps[-1].result_mixture.pdf(x) for x in x_linspace],
            color="green",
            label="estimated",
        )
        self.save_analysis()

    def overall_compare_methods(
        self,
        result_1: ExperimentDescription,
        result_2: ExperimentDescription,
        method_1: str,
        method_2: str,
    ):
        x_linspace = np.linspace(-10, 10, 1000)

        plt.hist(result_1.samples, color="lightsteelblue", density=True)
        plt.plot(
            x_linspace,
            [result_1.base_mixture.pdf(x) for x in x_linspace],
            color="red",
            label="base",
        )
        plt.plot(
            x_linspace,
            [result_1.steps[-1].result_mixture.pdf(x) for x in x_linspace],
            color="green",
            label=method_1,
        )
        plt.plot(
            x_linspace,
            [result_2.steps[-1].result_mixture.pdf(x) for x in x_linspace],
            color="purple",
            label=method_2,
            marker="+",
            markevery=range(1, 1000, 30),
        )
        self.save_analysis()
