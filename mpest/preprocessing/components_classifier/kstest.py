"""Module which contains Kolmogorov-Smirnov Test Classifier"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

from mpest.annotations import Samples
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_classifier.abstract_classifier import AComponentsClassifier
from mpest.preprocessing.utils import Model


class KSTest(AComponentsClassifier):
    """
    Kolmogorov-Smirnov Test Classifier
    -----
    :param k_init:        int         default: 1      — Number of times the KMeans is run
    :param k_max_iter:    int         default: 300    — Maximum number of iterations in KMeans
    :param random_state:  int | None  default: None   — Determines random generation parameters for mixture model
    """

    def __init__(
            self,
            k_init: int = 1,
            k_max_iter: int = 300,
            random_state: int | None = None
    ) -> None:
        self.k_init = k_init
        self.k_max_iter = k_max_iter
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "Kolmogorov-Smirnov Test Classifier"

    def predict(self, samples: Samples, k: int) -> list[Model]:

        def _choose_distribution(cluster: Samples) -> Model:
            distributions: list[Model] = [GaussianModel(), ExponentialModel(), WeibullModelExp()]
            results = []

            # Gaussian Distribution
            args_norm = stats.norm.fit(cluster)
            results.append(stats.kstest(cluster, "norm", args=args_norm)[0])

            # Exponential Distribution
            args_expon = stats.expon.fit(cluster)
            results.append(stats.kstest(cluster, "expon", args=args_expon)[0])

            # Weibull Distribution
            args_weibull = stats.weibull_min.fit(cluster)
            results.append(stats.kstest(cluster, "weibull_min", args=args_weibull)[0])

            return distributions[np.argmin(results)]

        kmeans = KMeans(
            max_iter=self.k_max_iter,
            n_clusters=k,
            init="k-means++",
            n_init=self.k_init,
            random_state=self.random_state,
        ).fit(samples.reshape(-1, 1))

        return [_choose_distribution(samples[kmeans.labels_ == i]) for i in range(k)]
