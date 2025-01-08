"""Module which contains Elbow Method"""

from kneed import KneeLocator
from sklearn.cluster import KMeans

from mpest.components_number.methods.abstract_estimator import AComponentsNumber
from mpest.types import Samples


class Elbow(AComponentsNumber):
    """
    Elbow method with KMeans++
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
        return "Elbow"

    def estimate(self, samples: Samples) -> int:
        samples = samples.reshape(-1, 1)
        k_range = range(1, self.kmax + 2)  # possible components: [2, kmax]
        wcss = []

        for k in k_range:
            kmeans_elbow = KMeans(
                max_iter=self.k_max_iter,
                n_clusters=k,
                init="k-means++",
                n_init=self.k_init,
                random_state=self.random_state,
            ).fit(samples)
            wcss.append(kmeans_elbow.inertia_)

        knee = KneeLocator(k_range, wcss, curve="convex", direction="decreasing").elbow

        return knee
