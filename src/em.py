from typing import NamedTuple
import functools
import warnings
import numpy as np

from optimizer import Optimizer, ScipyNewtonCG
from utils import Samples
from distribution import Distribution

warnings.filterwarnings("ignore")


class EM:
    def __init__(
        self,
        deviation: float = 0.01,
        max_step: int | None = None,
        prior_probability_threshold: float | None = None,
        prior_probability_threshold_step: int = 3,
        optimizer: type[Optimizer] = ScipyNewtonCG,
    ):
        self.deviation = deviation
        self.max_step = max_step
        self.prior_probability_threshold = prior_probability_threshold
        self.prior_probability_threshold_step = prior_probability_threshold_step
        self.optimizer = optimizer
        self.result = None

    # Describes all needed data about em algo result
    class Result(NamedTuple):
        distributions: list[Distribution]
        steps: int
        error: Exception | None = None

    # Static realization of em algo
    @staticmethod
    def em_algo(
        samples: Samples,
        distributions: list[Distribution],
        k: int,
        deviation: float = 0.01,
        max_step: int | None = 50,
        prior_probability_threshold: float | None = None,
        prior_probability_threshold_step: int = 3,
        optimizer: type[Optimizer] = ScipyNewtonCG,
    ) -> "EM.Result":
        step = 0

        class DistributionInProgress:
            def __init__(self, distribution: Distribution, ind: int):
                self.content = distribution
                self._is_active = True
                self.ind = ind

            def make_inactive(self):
                if (self.content.prior_probability is not None) and not np.isfinite(
                    self.content.prior_probability
                ):
                    new_prior_probability = self.content.prior_probability
                else:
                    new_prior_probability = None
                self.content = Distribution(
                    self.content.model, self.content.params, new_prior_probability
                )
                self._is_active = False

            def _check_if_finite(self) -> bool:
                if (self.content.prior_probability is not None) and not np.isfinite(
                    self.content.prior_probability
                ):
                    return False
                return bool(np.all(np.isfinite(self.content.params)))

            @property
            def is_active(self) -> bool:
                if not self._is_active:
                    return False
                if self.content.prior_probability is None:
                    self._is_active = False
                    return False
                if not self._check_if_finite():
                    self._is_active = False
                    return False
                if prior_probability_threshold is None:
                    return True
                if step < prior_probability_threshold_step:
                    return True
                if self.content.prior_probability < prior_probability_threshold:
                    self._is_active = False
                    return False
                return True

        class DistributionsInProgress:
            def __init__(self, distributions: list[Distribution]):
                self._active: list[DistributionInProgress] = []
                self._stopped: list[DistributionInProgress] = []
                for i, d in enumerate(distributions):
                    dip = DistributionInProgress(d, i)
                    if dip.is_active:
                        self._active.append(dip)
                    else:
                        self._stopped.append(dip)
                self._distributions_changed = True
                self._active_distributions = None

            @property
            def all_distributions(self) -> tuple[Distribution, ...]:
                return tuple(
                    [
                        d.content
                        for d in sorted(
                            self._active + self._stopped, key=lambda d: d.ind
                        )
                    ]
                )

            @property
            def active_distributions(self) -> tuple[Distribution, ...]:
                if self._distributions_changed or self._active_distributions is None:
                    self._active_distributions = self._update()
                return self._active_distributions

            def set_distribution(self, ind: int, distribution: Distribution) -> None:
                self._active[ind].content = distribution
                self._distributions_changed = True

            def _update(self) -> tuple[Distribution, ...]:
                new_active: list[DistributionInProgress] = []
                w_sum = 0
                for d in self._active:
                    if d.is_active:
                        new_active.append(d)
                        if d.content.prior_probability is not None:
                            w_sum += d.content.prior_probability
                    else:
                        self._stopped.append(d)
                        d.make_inactive()

                w_error = 1 - w_sum

                if (w_error > 0) and (len(new_active) > 0):
                    w_mean = w_error / len(new_active)
                    for d in new_active:
                        if d.content.prior_probability is not None:
                            d.content = Distribution(
                                d.content.model,
                                d.content.params,
                                d.content.prior_probability + w_mean,
                            )

                self._active = new_active
                return tuple([d.content for d in self._active])

        def end_cond(
            prev: tuple[Distribution, ...] | None, curr: tuple[Distribution, ...]
        ) -> bool:
            if (max_step is not None) and (step >= max_step):
                return False

            if prev is None:
                return True

            if len(prev) != len(curr):
                return True

            for d_p, d_c in zip(prev, curr):
                if np.any(np.abs(d_p.params - d_c.params) > deviation):
                    return True
                if np.any(
                    np.abs(d_p.prior_probability - d_c.prior_probability) > deviation
                ):
                    return True
            return False

        curr = DistributionsInProgress(
            [Distribution(model, o, 1 / k) for model, o, _ in distributions]
        )
        curr_a = curr.active_distributions
        prev = None

        while end_cond(prev, curr_a):
            prev = curr_a

            # E part

            # p_xij contain all non zero probabilities: p_xij = p(X_i | O_j) for each X_i
            p_xij = []
            a_samples = []
            for x in samples:
                p = np.array([model.p(x, o) for model, o, _ in curr_a])
                if np.any(p):
                    p_xij.append(p)
                    a_samples.append(x)

            if not a_samples:
                return EM.Result(
                    list(curr.all_distributions),
                    step,
                    Exception("All models can't match"),
                )

            # h[j, i] contains probability of X_i to be a part of distribution j
            m = len(p_xij)
            k = len(curr_a)
            h = np.zeros([k, m], dtype=float)
            curr_w = np.array([d.prior_probability for d in curr_a])
            for i, p in enumerate(p_xij):
                wp = curr_w * p
                swp = np.sum(wp)
                if not swp:
                    return EM.Result(
                        list(curr.all_distributions), step, Exception("Error in E step")
                    )
                h[:, i] = wp / swp

            # M part

            # Need attention due creating all w==np.nan problem
            # instead of removing distribution which is a cause of error
            new_w = np.sum(h, axis=1) / m

            for j, ch in enumerate(h[:]):
                model, o, _ = curr_a[j]

                def log_likelihood(o, ch, model):
                    return -np.sum(ch * [model.lp(x, o) for x in a_samples])

                def jacobian(o, ch, model):
                    return -np.sum(
                        ch
                        * np.swapaxes([model.ld_params(x, o) for x in a_samples], 0, 1),
                        axis=1,
                    )

                # maximizing log of likelihood function for every active distribution
                new_o = optimizer.minimize(
                    functools.partial(log_likelihood, ch=ch, model=model),
                    o,
                    jacobian=functools.partial(jacobian, ch=ch, model=model),
                )
                curr.set_distribution(j, Distribution(model, new_o, new_w[j]))

            curr_a = curr.active_distributions

            if len(curr_a) == 0:
                return EM.Result(
                    list(curr.all_distributions),
                    step,
                    Exception("All models can't match due prior probability"),
                )

            step += 1

        return EM.Result(list(curr.all_distributions), step)

    def fit(
        self,
        samples: Samples,
        distributions: list[Distribution],
        k: int,
    ) -> None:
        self.result = EM.em_algo(
            samples,
            distributions,
            k,
            deviation=self.deviation,
            max_step=self.max_step,
            prior_probability_threshold=self.prior_probability_threshold,
            prior_probability_threshold_step=self.prior_probability_threshold_step,
            optimizer=self.optimizer,
        )

    def predict(self, x):
        if not self.result:
            return 0
        return np.sum(
            [
                model.p(x, o) * w if w is not None else 0
                for model, o, w in self.result.distributions
            ]
        )
