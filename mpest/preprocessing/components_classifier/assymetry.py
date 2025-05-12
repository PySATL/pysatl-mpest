"""Module which contains Assymetry Classifier"""

from scipy.stats import skew
from sklearn.cluster import KMeans

from mpest.annotations import Samples
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_classifier.abstract_classifier import AComponentsClassifier
from mpest.preprocessing.utils import Model


class Assymetry(AComponentsClassifier):
    """
    Assymetry Classifier
    -----
    :param k_init:        int         default: 1      — Number of times the KMeans is run
    :param k_max_iter:    int         default: 300    — Maximum number of iterations in KMeans
    :param random_state:  int | None  default: None   — Determines random generation parameters for mixture model
    """

    def __init__(
        self,
        error_rate: float = 0.2,
        k_init: int = 1,
        k_max_iter: int = 300,
        random_state: int | None = None
    ) -> None:
        self.error_rate = error_rate
        self.k_init = k_init
        self.k_max_iter = k_max_iter
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "Assymetry Classifier"

    def predict(self, samples: Samples, k: int) -> list[Model]:

        def _choose_distribution(cluster: Samples) -> Model:
            cluster_skew = skew(cluster)

            if cluster_skew < self.error_rate:
                return GaussianModel()

            if cluster_skew > 2 - self.error_rate:
                return ExponentialModel()

            return WeibullModelExp()

        kmeans = KMeans(
            max_iter=self.k_max_iter,
            n_clusters=k,
            init="k-means++",
            n_init=self.k_init,
            random_state=self.random_state,
        ).fit(samples.reshape(-1, 1))

        return [_choose_distribution(samples[kmeans.labels_ == i]) for i in range(k)]
