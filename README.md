[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![CI](https://github.com/PySATL/pysatl-mpest/actions/workflows/check.yml/badge.svg)
<a href="https://github.com/PySATL/pysatl-mpest/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>

## Installation

The package is developed with [Poetry](https://python-poetry.org/). Clone the repository and install the dependencies:

```
poetry install
```

## Abstract

This package contains realization of EM-type algorithms for solving the parameter estimation of mixture distribution problem:

$$p(x | \Omega, F, \Theta) = \sum_{j=1}^k \omega_j f_j(x | \theta_j)$$

- $p(x | \Omega, F, \Theta)$ - mixture distribution
- $f_j(x | \theta_j) \in F$ - $j$ distribution
- $\omega_j \in \Omega$ - prior probability of $j$ distribution
- $\theta_j \in \Theta$ - parameters of $j$ distribution

The problem is to find $\Omega$ and $\Theta$ params of custom mixture distribution with known (or guessed) $k$, $F$ and guessed or randomized initial approximation.

The components of a mixture are distributions of [pysatl-core](https://github.com/PySATL/pysatl-core), so the package works with any family registered there, and a mixture may consist of different families, for example Gaussian and exponential ones. The package itself is concerned with the estimation algorithms and the optimizers they use.

## Usage

A component is created from a family that is registered in `pysatl-core`: `normal(mu=0.0, sigma=1.0)` is a normal distribution, and `normal.view(mu=0.0)(sigma=1.0)` is one whose mean is fixed and is not estimated. A `MixtureModel` holds the components and their weights. Importing `pysatl_mpest` registers the built-in families of `pysatl-core`.

The estimators fit a `MixtureModel` to a sample starting from an initial approximation:

- `EM` fits the mixture by the classic Expectation-Maximization algorithm: the E-step computes the responsibilities of the observations and the M-step updates the weights and every free parameter of each component by maximizing the Q-function.
- `ECM` maximizes the expected complete-data log-likelihood (the Q-function) separately for every component.
- `ECME` can maximize the observed data likelihood directly instead, for the components you choose.

All three take the same arguments:

- `breakpointers` - the stopping criteria. `StepBreakpointer(max_steps)` stops after a number of iterations, `LikelihoodBreakpointer(threshold)` stops when the log-likelihood no longer changes by more than the threshold.
- `pruners` - strategies that remove components during the fit. `PriorThresholdPruner(threshold)` removes the components whose weight falls below the threshold. May be empty.
- `optimizer` - a numerical optimizer for the parameters that have no closed-form estimate: `ScipyNelderMead` or `ScipyPowell`. For the normal (`meanStd`) and exponential (`rate`) parametrizations the estimates are computed analytically.

Note: the current `ECM` implementation performs the same component-wise Q-function updates as `EM`.

### Code example

```Python
from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import FamilyName

from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import EM
from pysatl_mpest.estimators.iterative import LikelihoodBreakpointer, PriorThresholdPruner, StepBreakpointer
from pysatl_mpest.optimizers import ScipyNelderMead

normal = ParametricFamilyRegister.get(FamilyName.NORMAL)


def show(mixture: MixtureModel) -> None:
    for weight, component in zip(mixture.weights, mixture):
        parameters = {name: round(value, 3) for name, value in component.parameters.items()}
        print(f"weight={weight:.3f}  {parameters}")


# A sample from a mixture of two normal distributions.
truth = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=6.0, sigma=1.5)], weights=[0.4, 0.6])
X = truth.generate(size=2000, random_state=0)

# The initial approximation: the number and the families of the components are known, the parameters are guessed.
start = MixtureModel([normal(mu=1.0, sigma=2.0), normal(mu=4.0, sigma=2.0)], weights=[0.5, 0.5])

em = EM(
    breakpointers=[LikelihoodBreakpointer(1e-6), StepBreakpointer(200)],
    pruners=[PriorThresholdPruner(0.05)],
    optimizer=ScipyNelderMead(),
)
show(em.fit(X, start))

# The mean of the first component is known: fix it, and only its standard deviation is estimated.
start = MixtureModel([normal.view(mu=0.0)(sigma=2.0), normal(mu=4.0, sigma=2.0)], weights=[0.5, 0.5])
show(em.fit(X, start))
```

Output:

```
weight=0.405  {'mu': -0.052, 'sigma': 0.972}
weight=0.595  {'mu': 5.978, 'sigma': 1.495}
weight=0.406  {'sigma': 0.98}
weight=0.594  {'mu': 5.984, 'sigma': 1.489}
```

`parameters` of a component lists only its free parameters, so the fixed mean of the first component is not shown in the second fit; it stays exactly `0.0`.

## Requirements
- python >= 3.12, < 3.14
- numpy
- scipy
- pysatl-core
