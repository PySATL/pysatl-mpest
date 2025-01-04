from pathlib import Path

from experimental_env.analysis.analysis import Analysis
from experimental_env.analysis.analyze_strategies.density_plot import DensityPlot
from experimental_env.analysis.analyze_strategies.error_convergence import (
    ErrorConvergence,
)
from experimental_env.analysis.analyze_strategies.time_plot import TimePlot
from experimental_env.analysis.analyze_summarizers.error_summarizer import (
    ErrorSummarizer,
)
from experimental_env.analysis.analyze_summarizers.time_summarizer import TimeSummarizer
from experimental_env.analysis.metrics import SquaredError
from experimental_env.experiment.experiment_parser import ExperimentParser

EXPERIMENT_DIR = "experiment"
WORKING_DIR = Path(
    f"/home/danil/PycharmProjects/Projects/EM-algo-DT/{EXPERIMENT_DIR}/stage_3"
)


# Compare results
LMOMENTS_DIR = Path(
    f"/home/danil/PycharmProjects/Projects/EM-algo-DT/{EXPERIMENT_DIR}/stage_2/LM-EM"
)
LIKELIHOOD_DIR = Path(
    f"/home/danil/PycharmProjects/Projects/EM-algo-DT/{EXPERIMENT_DIR}/stage_2/MLE-EM"
)

results_1 = ExperimentParser().parse(LMOMENTS_DIR)
results_2 = ExperimentParser().parse(LIKELIHOOD_DIR)

analyze_actions = [DensityPlot(), TimePlot(), ErrorConvergence(SquaredError())]
analyze_summarizers = [ErrorSummarizer(SquaredError()), TimeSummarizer()]

Analysis(WORKING_DIR, analyze_actions, analyze_summarizers).analyze(results_1, "LM-EM")
Analysis(WORKING_DIR, analyze_actions, analyze_summarizers).analyze(results_2, "MLE-EM")
Analysis(WORKING_DIR, analyze_actions, analyze_summarizers).compare(
    results_1, results_2, "LM-EM", "MLE-EM"
)
