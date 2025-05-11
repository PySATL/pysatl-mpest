"""The module in which the maximum likelihood method is presented"""

from functools import partial

import numpy as np
from scipy.stats import FitError, weibull_min
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem, Result
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.exceptions import SampleError
from mpest.models import AModel, AModelDifferentiable, WeibullModelExp
from mpest.optimizers import AOptimizerJacobian, TOptimizer
from mpest.utils import ResultWithError

EResult = tuple[list[float], np.ndarray, Problem] | ResultWithError[MixtureDistribution]


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
        samples = problem.samples
        mixture = problem.distributions
        p_xij = []
        active_samples = []
        for x in samples:
            p = np.array([d.model.pdf(x, d.params) for d in mixture])
            if np.any(p):
                p_xij.append(p)
                active_samples.append(x)

        if not active_samples:
            error = SampleError("None of the elements in the sample is correct for this mixture")
            return ResultWithError(mixture, error)

        # h[j, i] contains probability of X_i to be a part of distribution j
        m = len(p_xij)
        k = len(mixture)
        h = np.zeros([k, m], dtype=float)
        curr_w = np.array([d.prior_probability for d in mixture])
        for i, p in enumerate(p_xij):
            wp = curr_w * p
            swp = np.sum(wp)

            if not swp:
                return ResultWithError(mixture, ZeroDivisionError())

            h[:, i] = wp / swp

        return active_samples, h, problem


class ML(AExpectation[EResult]):
    """
    Improved ML initialization with proper distribution parameterization
    and tail handling to avoid divergence to -infinity
    """

    def __init__(self, models: list[AModel], method: str = "kmeans", eps: float = 0.3) -> None:
        self._n_components = len(models)
        self._method = method
        self._models = models
        self._initialized = False
        self._current_mixture = MixtureDistribution([])
        self._eps = eps

    @staticmethod
    def estimate_weibull_params(data: np.ndarray) -> list[float]:
        """Robust Weibull parameter estimation using MLE"""
        try:
            params = weibull_min.fit(data, floc=0)
            return [params[0], params[2]]
        except (ValueError, TypeError, FitError):
            return [0.5, np.mean(data)]

    def _initialize_distributions(self, X: np.ndarray) -> MixtureDistribution:
        """Improved initialization with distribution-aware parameter estimation"""
        if self._method == "kmeans":
            kmeans = KMeans(n_clusters=self._n_components)
            labels = kmeans.fit_predict(X.reshape(-1, 1))
        elif self._method == "dbscan":
            dbscan = DBSCAN(eps=self._eps, min_samples=5)
            labels = dbscan.fit_predict(X.reshape(-1, 1))
            if -1 in labels:
                labels[labels == -1] = np.random.choice(range(self._n_components), np.sum(labels == -1))
        elif self._method == "agglo":
            agglo = AgglomerativeClustering(n_clusters=self._n_components)
            labels = agglo.fit_predict(X.reshape(-1, 1))
        else:
            raise ValueError("Can't find this clustering method.")

        params_for_init = []
        weights: list[float | None]  = []

        for k in range(self._n_components):
            X_k = X[labels == k]
            weight = len(X_k) / len(X)

            if len(X_k) == 0:
                X_k = np.random.choice(X, size=10, replace=True)
                weight = 1.0 / self._n_components

            model = self._models[k]
            if isinstance(model, WeibullModelExp):
                params = self.estimate_weibull_params(X_k)
                params = list(np.clip(params, [0.1, 0.1], [2.0, 1000.0]))
                params[0], params[1] = float(params[0]), float(params[1])
            else:
                mean = np.mean(X_k)
                std = np.clip(np.std(X_k), 0.1, 100.0)
                params = [mean, std]

            params_for_init.append(params)
            weights.append(float(weight))

        self._current_mixture = MixtureDistribution.from_distributions(
            (
                [
                    Distribution.from_params(model.__class__, params)
                    for model, params in zip(self._models, params_for_init)
                ]
            ),
            weights
        )
        self._initialized = True
        return self._current_mixture

    def step(self, problem: Problem) -> EResult:
        """E-step with improved numerical stability"""
        if not self._initialized:
            mixture_dist = self._initialize_distributions(problem.samples)
        else:
            mixture_dist = problem.distributions
        samples = problem.samples

        p_xij = []
        active_samples = []

        min_prob = 1e-100

        for x in samples:
            p = np.zeros(len(mixture_dist.distributions))
            for i, d in enumerate(mixture_dist.distributions):
                try:
                    pdf_val = d.model.pdf(x, d.params)
                    p[i] = max(pdf_val, min_prob)
                except ValueError:
                    p[i] = min_prob

            if np.any(p > min_prob):
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

            if swp < min_prob:
                h[:, i] = curr_w / np.sum(curr_w)
            else:
                h[:, i] = wp / swp

        return active_samples, h, Problem(samples, mixture_dist)


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

        samples, h, problem = e_result
        optimizer = self.optimizer

        m = len(h[0])
        mixture = problem.distributions

        new_w = np.sum(h, axis=1) / m
        new_distributions: list[Distribution] = []
        for j, ch in enumerate(h[:]):
            d = mixture[j]

            def log_likelihood(params, ch, model: AModel):
                return -np.sum(ch * [model.lpdf(x, params) for x in samples])

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
