[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![CI](https://github.com/PySATL/pysatl-mpest/actions/workflows/check.yml/badge.svg)
<a href="https://github.com/PySATL/pysatl-mpest/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>

## Installation

The package is developed with [Poetry](https://python-poetry.org/). Clone the repository and install the dependencies:

```
poetry install
```

## Abstract

This package implements EM-type algorithms for estimating the parameters of a finite mixture of distributions:

$$f(x \mid \Psi) = \sum_{j=1}^k \omega_j f_j(x \mid \theta_j)$$

- $f(x \mid \Psi)$ - the mixture density
- $k$ - the number of components
- $f_j(x \mid \theta_j)$ - the density of the $j$-th component
- $\omega_j$ - the weight (prior probability) of the $j$-th component
- $\theta_j$ - the parameters of the $j$-th component
- $\Psi = (\omega_1, \dots, \omega_k, \theta_1, \dots, \theta_k)$ - all parameters of the mixture

The problem is to estimate $\Psi$ from a sample when $k$ and the families of the components are known (or guessed) and a user-provided initial approximation is given.

The components of a mixture are distributions of [pysatl-core](https://github.com/PySATL/pysatl-core), so the package works with any family registered there, and a mixture may consist of different families, for example Gaussian and exponential ones. The package itself is concerned with the estimation algorithms and the optimizers they use.

## Usage

A component is created from a family that is registered in `pysatl-core`: `normal(mu=0.0, sigma=1.0)` is a normal distribution, and `normal.view(mu=0.0)(sigma=1.0)` is one whose mean is fixed and is not estimated. A `MixtureModel` holds the components and their weights. Importing `pysatl_mpest` registers the built-in families of `pysatl-core`.

The estimators fit a `MixtureModel` to a sample starting from an initial approximation:

- `EM` fits the mixture by the classic Expectation-Maximization algorithm: the E-step computes the responsibilities of the observations and the M-step updates the weights and every free parameter of each component by maximizing the Q-function.
- `ECM` replaces the joint M-step of a component with conditional steps: the Q-function is maximized over ordered groups of the component's parameters, given by `parameter_groups` in `fit`, for example `parameter_groups={0: [{"sigma"}, {"mu"}]}`. Without `parameter_groups`, `ECM` is identical to `EM`.
- `ECME` chooses for every component whether to maximize the Q-function or the observed data likelihood directly. By default (`default_strategy="odl"`) every component maximizes the observed data likelihood; `q_indices_raw` and `odl_indices_raw` in `fit` choose the strategy per component.

All three take the same arguments:

- `breakpointers` - the stopping criteria. `StepBreakpointer(max_steps)` stops after a number of iterations, `LikelihoodBreakpointer(threshold)` stops when the log-likelihood no longer changes by more than the threshold.
- `pruners` - strategies that remove components during the fit. `PriorThresholdPruner(threshold)` removes the components whose weight falls below the threshold. May be empty.
- `optimizer` - a numerical optimizer for the parameters that have no closed-form estimate: `ScipyNelderMead` or `ScipyPowell`. For the normal (`meanStd`) and exponential (`rate`) parametrizations the estimates are computed analytically.

See the [estimator guide](docs/source/user_guide/estimators.md) for how to choose and configure an estimator.

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
