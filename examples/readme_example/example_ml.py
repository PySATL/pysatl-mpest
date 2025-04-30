import json
import os
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wasserstein_distance, entropy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from mpest import MixtureDistribution, Distribution, Problem
from mpest.em import EM
from mpest.em.breakpointers import StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker
from mpest.em.methods.likelihood_method import ML, BayesEStep, LikelihoodMStep
from mpest.em.methods.method import Method
from mpest.models import WeibullModelExp, GaussianModel
from mpest.optimizers import ScipyCOBYLA

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


class EnhancedML(ML):
    def __init__(self, models, n_components=1, method="kmeans", eps=None):
        super().__init__(models, n_components)
        self._method = method
        self.eps = eps

    def _get_labels(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Генерация меток в зависимости от метода"""
        X_reshaped = X.reshape(-1, 1)

        if self._method == "kmeans":
            kmeans = KMeans(n_clusters=self._n_components)
            return kmeans.fit_predict(X_reshaped)

        elif self._method == "dbscan":
            eps = self._auto_eps(X) if self.eps is None else self.eps
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_reshaped)
            return self._handle_noise(labels)

        elif self._method == "agglo":
            agglo = AgglomerativeClustering(n_clusters=self._n_components)
            return agglo.fit_predict(X_reshaped)

        return None

    def _auto_eps(self, X: np.ndarray, k: int = 5) -> float:
        """Автоподбор eps для DBSCAN"""
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(X.reshape(-1, 1))
        distances, _ = nbrs.kneighbors(X.reshape(-1, 1))
        return np.percentile(distances[:, -1], 95)

    def _handle_noise(self, labels: np.ndarray) -> np.ndarray:
        """Обработка шумовых точек"""
        if -1 in labels:
            labels[labels == -1] = max(labels) + 1
        return labels


def kl_divergence(true_mixture, fitted_mixture, x_min=0.001, x_max=10, n_points=1000):
    """Вычисление KL-дивергенции между истинным и подобранным распределениями"""
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
    """Расстояние Вассерштейна между распределениями"""
    samples_true = true_mixture.generate(n_points)
    samples_fit = fitted_mixture.generate(n_points)
    return wasserstein_distance(samples_true, samples_fit)


def evaluate_fit(true_mixture, fitted_mixture):
    """Оценка качества подгонки распределений"""
    return {
        'wasserstein': mixture_distance(true_mixture, fitted_mixture),
        'kl_divergence': kl_divergence(true_mixture, fitted_mixture)
    }


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Универсальная оценка качества кластеризации"""
    metrics = {
        'silhouette': -1,
        'calinski': -1,
        'davies_bouldin': np.inf
    }

    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        X_reshaped = X.reshape(-1, 1)
        metrics['silhouette'] = silhouette_score(X_reshaped, labels)
        metrics['calinski'] = calinski_harabasz_score(X_reshaped, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X_reshaped, labels)

    return metrics


def plot_results(ax, x, result, title, metrics=None):
    """Визуализация результатов с метриками"""
    sns.histplot(x, color="lightsteelblue", ax=ax)
    ax.set_xlabel("x")

    if metrics:
        metric_text = (f"\nSilhouette: {metrics.get('silhouette', 'N/A'):.2f}\n"
                       f"Calinski: {metrics.get('calinski', 'N/A'):.2f}\n"
                       f"Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.2f}\n"
                       f"Wasserstein: {metrics.get('wasserstein', 'N/A'):.2f}\n"
                       f"KL Divergence: {metrics.get('kl_divergence', 'N/A'):.2f}\n"
                       f"Time: {metrics.get('execution_time', 'N/A'):.2f}s")
        title += metric_text

    ax.set_title(title)

    ax_ = ax.twinx()
    ax_.set_ylabel("p(x)")
    ax_.set_yscale("log")

    X_plot = np.linspace(0.001, max(x), 3000)
    ax_.plot(X_plot, [base_mixture.pdf(x) for x in X_plot],
             color="green", label="True distribution")
    ax_.plot(X_plot, [result.result.pdf(x) for x in X_plot],
             color="red", label="Fitted distribution")
    ax_.legend()


def run_experiment(sample_size: int, results_dict: dict):
    """Запуск эксперимента для заданного размера выборки"""
    print(f"\nRunning experiment with sample size: {sample_size}")

    x = base_mixture.generate(sample_size)

    problem = Problem(
        x,
        MixtureDistribution.from_distributions(
            [
                Distribution.from_params(WeibullModelExp, [1.0, 2.0]),
                Distribution.from_params(GaussianModel, [0.0, 5.0]),
            ],
            [0.5, 0.5]
        ),
    )

    methods = [
        ("BayesEStep", None, BayesEStep()),
        ("KMeans+ML", "kmeans", EnhancedML([WeibullModelExp(), GaussianModel()],
                                           n_components=2, method="kmeans")),
        ("Agglo+ML", "agglo", EnhancedML([WeibullModelExp(), GaussianModel()],
                                         n_components=2, method="agglo")),
        ("DBSCAN+ML", "dbscan", EnhancedML([WeibullModelExp(), GaussianModel()],
                                           n_components=2, method="dbscan"))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Сравнение методов (n={sample_size})', fontsize=16)
    axes = axes.flatten()

    results = []
    for idx, (name, method_type, e_step) in enumerate(methods):
        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_step = LikelihoodMStep(ScipyCOBYLA())
            method = Method(e_step, m_step)
            em = EM(StepCountBreakpointer(max_step=128), FiniteChecker(), method=method)
            result = em.solve(problem)

        exec_time = time.time() - start_time
        metrics = {
            'execution_time': exec_time,
            **evaluate_fit(base_mixture, result.result)
        }

        if method_type:
            labels = e_step._get_labels(x)
            if labels is not None:
                metrics.update(evaluate_clustering(x, labels))
        else:
            metrics.update({
                'silhouette': np.nan,
                'calinski': np.nan,
                'davies_bouldin': np.nan
            })

        results.append((name, metrics))
        plot_results(axes[idx], x, result, name, metrics)

    plt.tight_layout()
    plt.savefig(f"results/plots/experiment_{sample_size}.png")
    plt.close()

    results_dict[sample_size] = []
    for name, metrics in results:
        results_dict[sample_size].append({
            'method': name,
            'metrics': metrics
        })

    print(f"\nResults for n={sample_size}:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Method", "Silhouette", "Calinski", "DB Index", "Wasserstein", "KL Diverg", "Time (s)"))

    for name, metrics in results:
        print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
            name,
            metrics.get('silhouette', np.nan),
            metrics.get('calinski', np.nan),
            metrics.get('davies_bouldin', np.nan),
            metrics['wasserstein'],
            metrics['kl_divergence'],
            metrics['execution_time']))


base_mixture = MixtureDistribution.from_distributions(
    [
        Distribution.from_params(WeibullModelExp, [0.5, 1.0]),
        Distribution.from_params(GaussianModel, [5.0, 1.0]),
    ],
    [0.33, 0.66],
)

results_data = {}
sample_sizes = [1000, 5000, 10000]

for size in sample_sizes:
    run_experiment(size, results_data)

with open("results/experiment_results.json", "w") as f:
    json.dump(results_data, f, indent=4)

print("\nВсе эксперименты завершены. Результаты сохранены в папке 'results'")