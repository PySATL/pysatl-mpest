"""The module in which the maximum likelihood method is presented"""

from functools import partial

import numpy as np
from scipy.stats import FitError, norm, weibull_min

from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem, Result
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.exceptions import EStepError, SampleError
from mpest.models import AModel, AModelDifferentiable, WeibullModelExp
from mpest.optimizers import AOptimizerJacobian, TOptimizer
from mpest.utils import ResultWithError

EResult = tuple[Problem, np.ndarray] | ResultWithError[MixtureDistribution]

class BayesEStep(AExpectation[EResult]):
    """
    Class which represents Bayesian method for calculating matrix for M step in likelihood method
    """


    def step(self, problem: Problem) -> EResult:
        """
        A function that performs E step

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Return active_samples, matrix with probabilities and problem.
        """

        samples = np.sort(problem.samples)
        mixture = problem.distributions

        p_xij = []
        active_samples = []

        for x in samples:
            p = np.array([d.model.pdf(x, d.params) for d in mixture])
            if np.any(p):
                p_xij.append(p)
                active_samples.append(x)

        if not active_samples:
            error = EStepError("None of the elements in the sample is correct for this mixture")
            return ResultWithError(mixture, error)

        # h[j, i] contains probability of X_i to be a part of distribution j
        m = len(p_xij)
        k = len(mixture)
        h = np.zeros([k, m], dtype=float) #matrix of hidden variables

        curr_weight = np.array([d.prior_probability for d in mixture])
        for i, p in enumerate(p_xij):
            wp = curr_weight * p
            swp = np.sum(wp)

            if not swp:
                return ResultWithError(mixture, ZeroDivisionError())
            h[:,  i] = wp / swp

        if np.isnan(h).any():
            return ResultWithError(problem.distributions, EStepError(""))

        new_problem = Problem(np.array(active_samples),
                              MixtureDistribution.from_distributions(mixture))

        return new_problem, h


class ClusteringEStep(AExpectation[EResult]):
    """
    E step that uses clustering methods for recalculate mixture parameters
    Supported methods: DBScan (dbscan), Agglomerative (agglo), KMeans (kmeans)
    Use accurate_init for the best accuracy of the
    parameter values for each individual component (recommended for mixtures from several different distributions)
    """
    MIN_SAMPLES = 2
    MIN_PROB = 1e-100
    MIN_COMPONENT_SIZE = 10
    def __init__(self,models: list[AModel], clusterizer,eps: float=0.3,accurate_init: bool=False) -> None:
        self._n_components = len(models)
        self._models = models
        self._initialized = False
        self._current_mixture = MixtureDistribution([])
        self._eps = eps
        self._accurate_init_flag = accurate_init
        self._clusterizer = clusterizer


    @staticmethod
    def _estimate_weibull_params(data: np.ndarray) -> list[float]:
        """Robust Weibull parameter estimation using MLE"""
        try:
            params = weibull_min.fit(data, floc=0)
            return [float(params[0]), float(params[2])]
        except (ValueError, TypeError, FitError):
            return [0.5, float(np.mean(data))]

    def _find_best_cluster_for_model(self, clusters: dict[int, np.ndarray],
                                     model: AModel) -> tuple[int | None, list[float] | None, float]:
        best_k, best_params, best_score = None, None, -np.inf
        for k, X_k in clusters.items():
            if len(X_k) < self.MIN_SAMPLES:
                continue
            try:
                if isinstance(model, WeibullModelExp):
                    params = self._estimate_weibull_params(X_k)
                    params_arr = np.clip(params, [0.1, 0.1], [2.0, 1000.0])
                    params = [float(params_arr[0]), float(params_arr[1])]
                    score = np.sum(np.log(weibull_min.pdf(X_k, *params)))
                else:
                    mean = np.mean(X_k)
                    std = np.clip(np.std(X_k), 0.1, 100.0)
                    params = [mean, std]
                    score = np.sum(np.log(norm.pdf(X_k, mean, std)))
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_params = params
            except ValueError:
                continue
        return best_k, best_params, best_score

    def _accurate_init(self, X: np.ndarray, labels: np.ndarray) -> tuple[list[tuple[AModel, list[float]]], list[float]]:
        clusters = {k: X[labels == k] for k in range(self._n_components)}
        distributions: list[tuple[AModel, list[float]]] = []
        weights: list[float] = []
        for model in self._models:
            best_k, best_params, best_score = self._find_best_cluster_for_model(clusters, model)

            if best_k is None or best_params is None:
                X_k = np.random.choice(X, size=10, replace=True)
                weight = 1.0 / self._n_components
                if isinstance(model, WeibullModelExp):
                    best_params = self._estimate_weibull_params(X_k)
                else:
                    best_params = [np.mean(X_k), np.std(X_k)]
            else:
                weight = len(clusters[best_k]) / len(X)
                clusters.pop(best_k)

            distributions.append(
                (model, best_params)
            )
            weights.append(float(weight))

        return distributions, weights

    def _fast_init(self, X: np.ndarray, labels: np.ndarray) -> tuple[list[tuple[AModel, list[float]]], list[float]]:
        distributions: list[tuple[AModel, list[float]]] = []
        weights: list[float] = []
        for k in range(self._n_components):
            X_k = X[labels == k]
            weight = len(X_k) / len(X)

            if len(X_k) == 0:
                X_k = np.random.choice(X, size=self.MIN_COMPONENT_SIZE, replace=True)
                weight = 1.0 / self._n_components

            model = self._models[k]
            if isinstance(model, WeibullModelExp):
                params = self._estimate_weibull_params(X_k)
                params = list(np.clip(params, [0.1, 0.1], [2.0, 1000.0]))
                params[0], params[1] = float(params[0]), float(params[1])
            else:
                mean = np.mean(X_k)
                std = np.clip(np.std(X_k), 0.1, 100.0)
                params = [mean, std]

            distributions.append((model, params))
            weights.append(float(weight))
        return distributions, weights

    def _initialize_distributions(self, X: np.ndarray, labels: np.ndarray) -> MixtureDistribution:
        """Improved initialization with distribution-aware parameter estimation"""
        if self._accurate_init_flag:
            distributions, weights = self._accurate_init(X, labels)
        else:
            distributions, weights = self._fast_init(X, labels)

        total_weight = sum(weights)
        normalized_weights: list[float | None] | None = [w / total_weight for w in weights]
        self._current_mixture = MixtureDistribution.from_distributions(
            (
                [
                    Distribution.from_params(dist[0].__class__, dist[1])
                    for dist in distributions
                ]
            ),
            normalized_weights
        )
        self._initialized = True
        return self._current_mixture

    def _clusterize(self, X: np.ndarray, clusterizer) -> np.ndarray:
        if hasattr(clusterizer, 'n_clusters') and self._n_components != clusterizer.n_clusters:
            raise EStepError("Count of components and clusters doesn't match.")
        X = X.reshape(-1, 1)
        labels = clusterizer.fit_predict(X)
        if -1 in labels:
            labels[labels == -1] = np.random.choice(range(self._n_components), np.sum(labels == -1))
        return labels


    def step(self, problem: Problem) -> EResult:
        """E-step with improved numerical stability"""
        samples = problem.samples
        if not self._initialized:
            try:
                labels = self._clusterize(samples, self._clusterizer)
                mixture_dist = self._initialize_distributions(samples, labels)
            except EStepError as e:
                return ResultWithError(problem.distributions, e)
        else:
            mixture_dist = problem.distributions

        p_xij = []
        active_samples = []

        for x in samples:
            p = np.zeros(len(mixture_dist.distributions))
            for i, d in enumerate(mixture_dist.distributions):
                try:
                    pdf_val = d.model.pdf(x, d.params)
                    p[i] = max(pdf_val, self.MIN_PROB)
                except ValueError:
                    p[i] = self.MIN_PROB

            if np.any(p > self.MIN_PROB):
                p_xij.append(p)
                active_samples.append(x)

        if not active_samples:
            error = SampleError("None of the elements in the sample is correct for this mixture")
            return ResultWithError(mixture_dist, error)

        m = len(p_xij)
        k = len(mixture_dist.distributions)
        h = np.zeros([k, m], dtype=float)
        curr_w = np.array([d.prior_probability or (1.0 / k) for d in mixture_dist.distributions])
        curr_w /= curr_w.sum()

        for i, p in enumerate(p_xij):
            wp = curr_w * p
            swp = np.sum(wp)

            if swp < self.MIN_PROB:
                h[:, i] = curr_w / np.sum(curr_w)
            else:
                h[:, i] = wp / swp

        new_problem = Problem(np.array(active_samples), mixture_dist)
        return new_problem, h


class LikelihoodMStep(AMaximization[EResult]):
    """
    Class which calculate new params using logarithm od likelihood function

    :param optimizer: The optimizer that is used in the step
    """

    def __init__(self, optimizer: TOptimizer):
        """
        Object constructor

        :param optimizer: The optimizer that is used in the step
        """
        self.optimizer = optimizer

    def step(self, e_result: EResult) -> Result:
        """
        A function that performs E step

        :param e_result: A tuple containing the arguments obtained from step E:
        active_samples, matrix with probabilities and problem.
        """

        if isinstance(e_result, ResultWithError):
            return e_result

        problem, h = e_result
        optimizer = self.optimizer

        m = len(h[0])
        samples = problem.samples
        mixture = problem.distributions

        new_w = np.sum(h, axis=1) / m
        new_distributions: list[Distribution] = []
        for j, ch in enumerate(h[:]):
            d = mixture[j]

            def log_likelihood(params, ch, model: AModel):
                array_ldpf = np.array([model.lpdf(x, params) for x in samples])
                finite_mask = np.isfinite(array_ldpf)
                return -np.sum(ch[finite_mask] * array_ldpf[finite_mask])

            def jacobian(params, ch, model: AModelDifferentiable):
                return -np.sum(
                    ch * np.swapaxes([model.ld_params(x, params) for x in samples], 0, 1),
                    axis=1,
                )

            # maximizing log of likelihood function for every active distribution
            if isinstance(optimizer, AOptimizerJacobian):
                if not isinstance(d.model, AModelDifferentiable):
                    raise TypeError

                new_params = optimizer.minimize(
                    partial(log_likelihood, ch=ch, model=d.model),
                    d.params,
                    jacobian=partial(jacobian, ch=ch, model=d.model),
                )
            else:
                new_params = optimizer.minimize(
                    func=partial(log_likelihood, ch=ch, model=d.model),
                    params=d.params,
                )

            new_distributions.append(Distribution(d.model, new_params))
        return ResultWithError(MixtureDistribution.from_distributions(new_distributions, new_w))
