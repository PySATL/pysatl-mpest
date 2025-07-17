import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from mpest import Distribution, MixtureDistribution, Problem
from mpest.em import EM
from mpest.em.breakpointers import StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker
from mpest.em.methods.likelihood_method import BayesEStep, ClusteringEStep, LikelihoodMStep
from mpest.em.methods.method import Method
from mpest.models import GaussianModel, WeibullModelExp
from mpest.optimizers import ScipyCG

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/plots/comparison", exist_ok=True)
os.makedirs("results/plots/pairs", exist_ok=True)


class EnhancedClusteringEStep(ClusteringEStep):
    def __init__(self, models, clusterizer, method="kmeans", eps=None):
        super().__init__(models, clusterizer)
        self._method = method
        self.eps = eps

    def get_labels(self, X: np.ndarray) -> np.ndarray | None:
        X_reshaped = X.reshape(-1, 1)
        if self._method == "kmeans":
            kmeans = KMeans(n_clusters=self._n_components)
            return kmeans.fit_predict(X_reshaped)
        elif self._method == "dbscan":
            eps = self.auto_eps(X) if self.eps is None else self.eps
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_reshaped)
            return self._handle_noise(labels)
        elif self._method == "agglo":
            agglo = AgglomerativeClustering(n_clusters=self._n_components)
            return agglo.fit_predict(X_reshaped)
        return None

    @staticmethod
    def auto_eps(X: np.ndarray, k: int = 5, percentile: float = 0.6) -> float:
        if len(X) < k + 1:
            raise ValueError(f"Need at least {k + 1} points to compute {k}-nearest neighbors")
        X_2d = X.reshape(-1, 1)
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(X_2d)
        distances, _ = nbrs.kneighbors(X_2d)
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)
        eps = np.percentile(k_distances_sorted, percentile)
        return float(eps)

    @staticmethod
    def _handle_noise(labels: np.ndarray) -> np.ndarray:
        if -1 in labels:
            labels[labels == -1] = max(labels) + 1
        return labels


def kl_divergence(true_mixture, fitted_mixture, x_min=0.001, x_max=10, n_points=1000):
    x = np.linspace(x_min, x_max, n_points)
    p = np.array([true_mixture.pdf(xi) for xi in x])
    q = np.array([fitted_mixture.pdf(xi) for xi in x])
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return entropy(p, q)


def mixture_distance(true_mixture, fitted_mixture, n_points: int = 1000) -> float:
    samples_true = true_mixture.generate(n_points)
    samples_fit = fitted_mixture.generate(n_points)
    return wasserstein_distance(samples_true, samples_fit)


def evaluate_fit(true_mixture, fitted_mixture):
    return {
        "wasserstein": mixture_distance(true_mixture, fitted_mixture),
        "kl_divergence": kl_divergence(true_mixture, fitted_mixture),
    }


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    metrics = {"silhouette": -1, "calinski": -1, "davies_bouldin": np.inf}
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        X_reshaped = X.reshape(-1, 1)
        metrics["silhouette"] = silhouette_score(X_reshaped, labels)
        metrics["calinski"] = calinski_harabasz_score(X_reshaped, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X_reshaped, labels)
    return metrics


def plot_distributions(ax, x, true_mixture, fitted_mixture, title):
    label_fontsize = 16
    title_fontsize = 18
    legend_fontsize = 14
    tick_fontsize = 14

    sns.histplot(x, color="royalblue", ax=ax, stat="density", alpha=0.8, binwidth=0.5, edgecolor="white", linewidth=1)

    ax.set_xlabel("Значение x", fontsize=label_fontsize, fontweight="bold", labelpad=10)
    ax.set_ylabel("Плотность (density)", fontsize=label_fontsize, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlim(0, 20)

    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_yticks(np.linspace(0, ax.get_yticks().max(), len(ax.get_yticks())))

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=tick_fontsize,
        width=3,
        length=8,
        pad=8,
        colors="black",
        grid_color="black",
        grid_alpha=0.5,
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

    ax_ = ax.twinx()
    ax_.set_ylabel("p(x)", fontsize=label_fontsize, fontweight="bold", labelpad=15)
    ax_.set_yscale("log")

    y_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax_.set_yticks(y_ticks)
    ax_.set_yticklabels([f"{tick:.2f}" for tick in y_ticks], fontsize=tick_fontsize, fontweight="bold", color="black")

    ax_.tick_params(axis="y", which="both", width=3, length=8, pad=10, colors="black")

    ax_.set_ylim(bottom=y_ticks[0], top=y_ticks[-1])

    for spine in ax_.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

    X_plot = np.linspace(0.001, 20, 1000)
    ax_.plot(
        X_plot,
        [true_mixture.pdf(xi) for xi in X_plot],
        color="darkgreen",
        label="Истинное распределение",
        linewidth=4,
        linestyle="-",
        alpha=0.9,
    )
    ax_.plot(
        X_plot,
        [fitted_mixture.pdf(xi) for xi in X_plot],
        color="crimson",
        label="Подобранное распределение",
        linewidth=4,
        linestyle="--",
        alpha=0.9,
    )

    legend = ax_.legend(
        loc="upper right",
        fontsize=legend_fontsize,
        framealpha=1,
        edgecolor="black",
        facecolor="white",
        frameon=True,
        borderpad=1,
    )
    legend.get_frame().set_linewidth(2)

    ax.minorticks_on()
    ax_.minorticks_on()
    ax.tick_params(axis="both", which="minor", width=2, length=5)
    ax_.tick_params(axis="both", which="minor", width=2, length=5)

    for y in y_ticks:
        ax_.axhline(y=y, color="gray", linestyle=":", alpha=0.3, linewidth=1)


def save_metrics_table(metrics_data: dict[str, dict[str, float]], filename: str, title: str):
    """Save metrics table with mean and std values"""
    df = pd.DataFrame.from_dict(metrics_data, orient="index")

    columns_order = []
    for metric in ["silhouette", "calinski", "davies_bouldin", "wasserstein", "kl_divergence", "execution_time"]:
        if metric in df.columns:
            columns_order.append(metric)
            if f"{metric}_std" in df.columns:
                columns_order.append(f"{metric}_std")

    df = df[columns_order]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    for method, row in df.iterrows():
        table_row = [method]
        for col in df.columns:
            if "_std" in col:
                continue
            mean = row[col]
            std = row.get(f"{col}_std", np.nan)
            table_row.append(f"{mean:.2f} ± {std:.2f}")
        table_data.append(table_row)

    table = ax.table(
        cellText=table_data,
        colLabels=["Method"] + [col for col in df.columns if "_std" not in col],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title(title, y=1.08)
    plt.savefig(f"results/tables/{filename}.png", bbox_inches="tight", dpi=300)
    plt.close()

    df.to_csv(f"results/tables/{filename}.csv")


def _initialize_methods(mixture: MixtureDistribution, eps) -> list[tuple]:
    """Initialize all methods to be tested"""
    models = []
    for dist in mixture.distributions:
        model_type = type(dist.model)
        if model_type == WeibullModelExp:
            models.append(WeibullModelExp())
        elif model_type == GaussianModel:
            models.append(GaussianModel())
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    n_clusters = len(models)
    return [
        ("BayesEStep", None, BayesEStep()),
        ("KMeans+ML", "kmeans", EnhancedClusteringEStep(models, clusterizer=KMeans(n_clusters=n_clusters))),
        (
            "Agglo+ML",
            "agglo",
            EnhancedClusteringEStep(models, clusterizer=AgglomerativeClustering(n_clusters=n_clusters)),
        ),
        ("DBSCAN+ML", "dbscan", EnhancedClusteringEStep(models, eps=eps, clusterizer=DBSCAN())),
    ]


def _run_em_method(problem: Problem, e_step, mixture: MixtureDistribution) -> dict:
    """Run a single EM method and return metrics"""
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_step = LikelihoodMStep(ScipyCG())
        method = Method(e_step, m_step)
        em = EM(StepCountBreakpointer(max_step=128), FiniteChecker(), method=method)
        result = em.solve(problem)

    exec_time = time.time() - start_time
    metrics = {"execution_time": exec_time, **evaluate_fit(mixture, result.result)}

    return metrics


def _calculate_clustering_metrics(x: np.ndarray, e_step, method_type: str, metrics: dict) -> dict:
    """Calculate and add clustering metrics if applicable"""
    if method_type:
        labels = e_step.get_labels(x)
        if labels is not None:
            metrics.update(evaluate_clustering(x, labels))
    else:
        metrics.update({"silhouette": np.nan, "calinski": np.nan, "davies_bouldin": np.nan})
    return metrics


def _calculate_summary_metrics(all_results: dict) -> dict:
    """Calculate mean and std of metrics across all runs"""
    summary_metrics = {}
    for method, runs in all_results.items():
        method_metrics = {}
        for key in runs[0]:
            values = [run[key] for run in runs]
            method_metrics[key] = float(np.nanmean(values))
            method_metrics[f"{key}_std"] = float(np.nanstd(values))
        summary_metrics[method] = method_metrics
    return summary_metrics


def _save_comparison_plots(
    methods: list,
    mixture: MixtureDistribution,
    problem: Problem,
    summary_metrics: dict,
    group_name: str,
    sample_size: int,
):
    """Save all comparison plots with metrics under titles"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    # fig.suptitle(f"Comparison of methods for {group_name} group (n={sample_size})", fontsize=16)
    axes = axes.flatten()

    for idx, (name, method_type, e_step) in enumerate(methods):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_step = LikelihoodMStep(ScipyCG())
            method = Method(e_step, m_step)
            em = EM(StepCountBreakpointer(max_step=128), FiniteChecker(), method=method)
            result = em.solve(problem)

        # metrics = summary_metrics[name]

        title = (
            f"{name}"
            # f"{name}\n\n"
            # f"Silhouette: {metrics['silhouette']:.3f} ± {metrics['silhouette_std']:.3f}\n"
            # f"Calinski-Harabasz: {metrics['calinski']:.1f} ± {metrics['calinski_std']:.1f}\n"
            # f"Davies-Bouldin: {metrics['davies_bouldin']:.3f} ± {metrics['davies_bouldin_std']:.3f}\n"
            # f"Wasserstein: {metrics['wasserstein']:.3f} ± {metrics['wasserstein_std']:.3f}\n"
            # f"KL Divergence: {metrics['kl_divergence']:.3f} ± {metrics['kl_divergence_std']:.3f}\n"
            # f"Time: {metrics['execution_time']:.1f}s ± {metrics['execution_time_std']:.1f}"
        )

        plot_distributions(axes[idx], problem.samples, mixture, result.result, title)

    plt.tight_layout()
    plt.savefig(f"results/plots/comparison/comparison_{group_name}.png", dpi=300)
    plt.close()

    _save_pair_plots(methods, mixture, problem, group_name)


def _save_pair_plots(methods: list, mixture: MixtureDistribution, problem: Problem, group_name: str):
    """Save pair comparison plots with metrics"""
    # Bayes vs KMeans
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Bayes vs KMeans for {group_name} group", fontsize=16)

    for ax, (name, _, e_step) in zip(axes, methods[:2]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_step = LikelihoodMStep(ScipyCG())
            method = Method(e_step, m_step)
            em = EM(StepCountBreakpointer(max_step=128), FiniteChecker(), method=method)
            result = em.solve(problem)

        title = f"{name}"
        plot_distributions(ax, problem.samples, mixture, result.result, title)

    plt.tight_layout()
    plt.savefig(f"results/plots/pairs/bayes_kmeans_{group_name}.png", dpi=300)
    plt.close()

    # DBSCAN vs Agglo
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"DBSCAN vs Agglo for {group_name} group", fontsize=16)

    for ax, (name, _, e_step) in zip(axes, methods[2:]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_step = LikelihoodMStep(ScipyCG())
            method = Method(e_step, m_step)
            em = EM(StepCountBreakpointer(max_step=128), FiniteChecker(), method=method)
            result = em.solve(problem)
        title = f"{name}"
        plot_distributions(ax, problem.samples, mixture, result.result, title)

    plt.tight_layout()
    plt.savefig(f"results/plots/pairs/dbscan_agglo_{group_name}.png", dpi=300)
    plt.close()


def run_experiment_group(
    mixture: MixtureDistribution, sample_size: int, n_experiments: int = 5, group_name: str = "default"
) -> dict[str, dict[str, float]]:
    """Run multiple experiments for a given mixture model"""
    all_results = {method: [] for method in ["BayesEStep", "KMeans+ML", "Agglo+ML", "DBSCAN+ML"]}

    for exp_num in range(n_experiments):
        print(f"Running experiment {exp_num + 1}/{n_experiments} for {group_name} group")
        x = mixture.generate(sample_size)
        eps = EnhancedClusteringEStep.auto_eps(x)
        problem = Problem(x, mixture)
        methods = _initialize_methods(mixture, eps)

        for name, method_type, e_step in methods:
            metrics = _run_em_method(problem, e_step, mixture)
            metrics = _calculate_clustering_metrics(x, e_step, method_type, metrics)
            all_results[name].append(metrics)

    summary_metrics = _calculate_summary_metrics(all_results)
    save_metrics_table(
        summary_metrics,
        f"metrics_{group_name}",
        f"Metrics for {group_name} group (n={sample_size}, {n_experiments} runs)",
    )

    x = mixture.generate(sample_size)
    eps = EnhancedClusteringEStep.auto_eps(x)
    problem = Problem(x, mixture)
    _save_comparison_plots(
        _initialize_methods(mixture, eps), mixture, problem, summary_metrics, group_name, sample_size
    )

    return summary_metrics


base_mixture = MixtureDistribution.from_distributions(
    [
        Distribution.from_params(WeibullModelExp, [0.5, 1.0]),
        Distribution.from_params(GaussianModel, [5.0, 1.0]),
    ],
    [0.33, 0.66],
)

gaussian_mixture = MixtureDistribution.from_distributions(
    [
        Distribution.from_params(GaussianModel, [0.0, 1.0]),
        Distribution.from_params(GaussianModel, [5.0, 1.0]),
    ],
    [0.5, 0.5],
)

weibull_mixture = MixtureDistribution.from_distributions(
    [
        Distribution.from_params(WeibullModelExp, [1.0, 1.0]),
        Distribution.from_params(WeibullModelExp, [5.0, 50.0]),
    ],
    [0.5, 0.5],
)

results_data = {}
sample_size = 1000

# 1. Base experiment (Weibull + Gaussian)
print("\nRunning experiments for based mixture (Weibull + Gaussian)")
results_data["original"] = run_experiment_group(base_mixture, sample_size, 5, "original")

# 2. Distant Gaussians
print("\nRunning experiments for distant Gaussians")
results_data["gaussian"] = run_experiment_group(gaussian_mixture, sample_size, 5, "gaussian")

# 3. Distant Weibulls
print("\nRunning experiments for distant Weibulls")
results_data["weibull"] = run_experiment_group(weibull_mixture, sample_size, 5, "weibull")

with open("results/experiment_results.json", "w") as f:
    json.dump(results_data, f, indent=4)

first_exp_metrics = results_data["original"]
summary_metrics = {
    method: {
        "silhouette": metrics["silhouette"],
        "silhouette_std": metrics["silhouette_std"],
        "kl_divergence": metrics["kl_divergence"],
        "kl_divergence_std": metrics["kl_divergence_std"],
        "execution_time": metrics["execution_time"],
        "execution_time_std": metrics["execution_time_std"],
    }
    for method, metrics in first_exp_metrics.items()
}

save_metrics_table(
    summary_metrics, "summary_first_experiment", "Summary metrics for first experiment (original mixture)"
)

print("\nAll experiments completed. Results saved in 'results' directory")
