"""Module which contains Silhouette Method"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mpest.components_number.methods.abstract_estimator import AComponentsNumber
from mpest.types import Samples


class Silhouette(AComponentsNumber):
    """
    Silhouette method with KMeans++
    -----
    :param kmax:       int                       — Assumed maximum number of components
    :param k_init:     int         default: 1    — Number of times the KMeans is run
    :param k_max_iter: int         default: 300  — Maximum number of iterations in KMeans
    :random_state:     int | None  default: None — Determines random generation for KMeans
    """

    def __init__(
        self,
        kmax: int,
        k_init: int = 1,
        k_max_iter: int = 300,
        random_state: int | None = None,
    ) -> None:
        self.kmax = kmax
        self.k_init = k_init
        self.k_max_iter = k_max_iter
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "Silhouette"

    def estimate(self, samples: Samples) -> int:
        samples = samples.reshape(-1, 1)
        k_range = range(2, self.kmax + 1)  # possible components: [2, kmax]
        silhouettes = []

        for k in k_range:
            kmeans_silhouette = KMeans(
                n_clusters=k,
                max_iter=self.k_max_iter,
                init="k-means++",
                n_init=self.k_init,
                random_state=self.random_state,
            ).fit(samples)
            silhouettes.append(silhouette_score(samples, kmeans_silhouette.labels_))

        optimal_k = silhouettes.index(max(silhouettes)) + 2

        return optimal_k
