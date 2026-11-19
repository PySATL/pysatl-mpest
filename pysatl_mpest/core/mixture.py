"""Finite mixture of continuous probability distributions."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.transformations.operations.distributions.mixture import FiniteMixtureDistribution, finite_mixture
from scipy.special import log_softmax, logsumexp, softmax

from ..typings import FloatArray, Scalar, UnivariateFloatArray
from . import _components
from ._sampling import DeterministicSampling

# Sampling does not depend on the sampling strategy of a component: pysatl-core's default one ignores ``random_state``.
_SAMPLING = DeterministicSampling()


def _normalize_log_weights(log_weights: ArrayLike, n_components: int) -> UnivariateFloatArray:
    """Copy log-weights and normalize them via log-softmax.

    Parameters
    ----------
    log_weights : ArrayLike
        Logarithmic component weights; they are normalized, so they need not sum to one in log space.
    n_components : int
        Expected number of weights.

    Returns
    -------
    UnivariateFloatArray
        A read-only normalized log-weights array of length ``n_components``.

    Raises
    ------
    ValueError
        If the length differs from ``n_components`` or the values cannot be normalized
        (all ``-inf``, ``NaN`` or ``+inf``).
    """

    new_log_weights = np.array(log_weights, dtype=np.float64)

    if len(new_log_weights) != n_components:
        raise ValueError(
            f"The length of the new log-weights vector ({len(new_log_weights)}) "
            f"does not match the number of components ({n_components})."
        )
    if not np.isfinite(logsumexp(new_log_weights)):
        raise ValueError("The log-weights must be normalizable: at least one finite value and no NaN or +inf.")

    normalized = log_softmax(new_log_weights)
    normalized.flags.writeable = False
    return normalized


class MixtureModel:
    """Represents a finite mixture of continuous probability distributions.

    This class encapsulates a collection of distribution components and their
    corresponding weights.

    Instances of this class can be compared for equality (``==``) and
    inequality (``!=``). Two models are considered equal if they have the
    same set of components and weights, regardless of the order in which
    components were added.

    The model is immutable: its components and weights cannot be changed after
    creation. Every modification returns a new ``MixtureModel`` via one of the
    ``with_*`` methods or :meth:`without_component`. Because the contents never
    change in place, instances are hashable and can safely be used as
    dictionary keys.

    Parameters
    ----------
    components : Sequence[ParametricFamilyDistribution]
        A sequence of distribution objects that will form the mixture.
    weights : ArrayLike | None, optional
        An array of initial weights for the components. The weights must be
        non-negative and sum to 1. If None, components are assigned equal
        weights. Defaults to None.

    Attributes
    ----------
    components : tuple[ParametricFamilyDistribution, ...]
        A tuple of the distribution objects that form the mixture.
    n_components : int
        The number of components in the mixture.
    weights : UnivariateFloatArray
        A NumPy array of the normalized weights for each component. The sum
        of weights is always 1.
    log_weights : UnivariateFloatArray
        A read-only NumPy array of normalized logarithmic component weights. New weights are set
        via :meth:`with_log_weights`.

    Raises
    ------
    ValueError
        If the list of components is empty, or if the provided weights are
        invalid (e.g., negative, wrong number of elements, or do not sum to 1).

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        with_component
        with_components
        with_log_weights
        with_added_component
        without_component
        to_core_mixture
        pdf
        lpdf
        component_lpdf_matrix
        loglikelihood
        generate
    """

    def __init__(
        self,
        components: Sequence[ParametricFamilyDistribution],
        weights: ArrayLike | None = None,
    ):
        n_components = len(components)
        if n_components == 0:
            raise ValueError("List of components cannot be empty")

        if weights is None:
            weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            self._validate_weights(n_components, weights)

        self._components = tuple(components)
        self._cached_weights: UnivariateFloatArray | None = None
        self._sorted_pairs_cache: list[tuple[_components.ComponentKey, float]] | None = None
        self._log_weights = _normalize_log_weights(np.log(weights + np.finfo(np.float64).tiny), n_components)

    @property
    def n_components(self):
        """int: Number of components in the mixture model."""

        return len(self.components)

    @property
    def components(self) -> tuple[ParametricFamilyDistribution, ...]:
        """tuple[ParametricFamilyDistribution, ...]: Components of the mixture."""

        return self._components

    @property
    def weights(self) -> UnivariateFloatArray:
        """UnivariateFloatArray: Normalized component weights cached from log-weights; treated as read-only."""

        if self._cached_weights is None:
            self._cached_weights = softmax(self._log_weights)

        return self._cached_weights  # type: ignore

    @property
    def log_weights(self) -> UnivariateFloatArray:
        """UnivariateFloatArray: Read-only normalized logarithmic component weights."""

        return self._log_weights

    def with_component(self, index: int, component: ParametricFamilyDistribution) -> "MixtureModel":
        """Return a new mixture with the component at ``index`` replaced.

        The log-weights are unchanged and shared with this mixture.

        Parameters
        ----------
        index : int
            Index of the component to replace; negative indices are not supported.
        component : ParametricFamilyDistribution
            The replacement component.

        Returns
        -------
        MixtureModel
            A new mixture; this mixture is left unchanged.

        Raises
        ------
        IndexError
            If ``index`` is outside ``[0, n_components - 1]``.
        """

        if not (0 <= index <= self.n_components - 1):
            raise IndexError(f"Index of component must be between 0 and {self.n_components - 1}")

        components = list(self._components)
        components[index] = component
        return self._from_normalized(components, self._log_weights)

    def with_components(self, components: Sequence[ParametricFamilyDistribution]) -> "MixtureModel":
        """Return a new mixture with all components replaced.

        The already normalized log-weights are kept as is and shared with this
        mixture; only the length is checked.

        Parameters
        ----------
        components : Sequence[ParametricFamilyDistribution]
            The new components; must be as many as in this mixture.

        Returns
        -------
        MixtureModel
            A new mixture; this mixture is left unchanged.

        Raises
        ------
        ValueError
            If the number of components differs from ``n_components``.
        """

        if len(components) != self.n_components:
            raise ValueError(f"Expected {self.n_components} components, got {len(components)}.")

        return self._from_normalized(components, self._log_weights)

    def with_log_weights(self, log_weights: ArrayLike) -> "MixtureModel":
        """Return a new mixture with the given log-weights.

        The input is copied and normalized via log-softmax, exactly as the
        weights passed to the constructor.

        Parameters
        ----------
        log_weights : ArrayLike
            New logarithmic component weights; they are normalized, so they need
            not sum to one in log space.

        Returns
        -------
        MixtureModel
            A new mixture; this mixture is left unchanged.

        Raises
        ------
        ValueError
            If the length differs from ``n_components`` or the values cannot be
            normalized (all ``-inf``, ``NaN`` or ``+inf``).
        """

        normalized = _normalize_log_weights(log_weights, self.n_components)
        return self._from_normalized(self._components, normalized)

    def with_added_component(self, component: ParametricFamilyDistribution, weight: Scalar) -> "MixtureModel":
        """Return a new mixture with an appended component.

        The existing component weights are multiplied by ``1 - weight`` so that
        their proportions are preserved.

        Parameters
        ----------
        component : ParametricFamilyDistribution
            The distribution component to add.
        weight : Scalar
            The weight for the new component, a number in the range (0, 1).

        Returns
        -------
        MixtureModel
            A new mixture; this mixture is left unchanged.

        Raises
        ------
        ValueError
            If ``weight`` is outside the range (0, 1).
        """

        if not (0 < weight < 1):
            raise ValueError("The weight of the new component must be in the range (0, 1).")

        d_weight = float(weight)
        new_log_weights = np.append(self._log_weights + np.log(1.0 - d_weight), np.log(d_weight))
        normalized = _normalize_log_weights(new_log_weights, self.n_components + 1)
        return self._from_normalized([*self._components, component], normalized)

    def without_component(self, index: int) -> "MixtureModel":
        """Return a new mixture without the component at ``index``.

        The weights of the remaining components are renormalized to sum to 1.

        Parameters
        ----------
        index : int
            The index of the component to remove; negative indices are not supported.

        Returns
        -------
        MixtureModel
            A new mixture; this mixture is left unchanged.

        Raises
        ------
        IndexError
            If the component index is out of range.
        ValueError
            If an attempt is made to remove the last component or the only
            component with positive weight.
        """

        n_components = self.n_components
        if not 0 <= index < n_components:
            raise IndexError(f"Index {index} out of range [0, {n_components - 1}]")

        if n_components <= 1:
            raise ValueError("The last component cannot be removed from the model.")

        new_log_weights = np.delete(self._log_weights, index)
        if not np.any(np.isfinite(new_log_weights)):
            raise ValueError("Cannot remove the only component with positive weight.")

        normalized = _normalize_log_weights(new_log_weights, n_components - 1)
        components = [comp for i, comp in enumerate(self._components) if i != index]
        return self._from_normalized(components, normalized)

    def to_core_mixture(self) -> FiniteMixtureDistribution:
        """Convert to a pysatl-core FiniteMixtureDistribution.

        The result is an independent pysatl-core mixture built from the current components and weights.
        """

        weighted_components = [(float(w), comp) for w, comp in zip(self.weights, self.components)]
        return finite_mixture(weighted_components)

    def pdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """Probability Density Function of the mixture: ``exp(lpdf(X))``.

        The shape of the result and the handling of scalar input follow :meth:`lpdf`.
        """

        return np.exp(self.lpdf(X))

    def lpdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """Logarithm of the Probability Density Function of the mixture.

        The result has the shape of ``X``; a scalar ``X`` gives a ``np.float64``. Points outside the
        support of every component evaluate to ``-inf``.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)
        result = logsumexp(self.component_lpdf_matrix(X) + self.log_weights, axis=-1)

        if is_scalar:
            return np.float64(result[()])
        return np.asarray(result, dtype=np.float64)

    def component_lpdf_matrix(self, X: ArrayLike) -> FloatArray:
        """Log-densities of every component, without the mixture weights.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        FloatArray
            An array of shape ``(*X.shape, n_components)`` whose last axis indexes
            the components. For a sample of ``n`` points this is an ``(n, n_components)`` matrix.
            The log-densities of one component are contiguous in memory (the matrix is a view of a
            ``(n_components, *X.shape)`` array), which makes reductions over the component axis,
            such as ``logsumexp`` in the E-step, several times faster than on a row-major matrix.
        """

        X = np.asarray(X, dtype=np.float64)
        stacked = np.stack([_components.lpdf(component, X) for component in self.components])
        return np.moveaxis(stacked, 0, -1)

    def loglikelihood(self, X: ArrayLike) -> float:
        """Sum of log-densities for all points in ``X``.

        The sum is ``-inf`` if any point is outside the support of every component.
        """

        X = np.asarray(X, dtype=np.float64)
        return np.sum(self.lpdf(X))

    def generate(
        self,
        size: int | np.integer[np.typing.NBitBase] | tuple[int, ...] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.float64 | FloatArray:
        """Sample random variates from the mixture distribution.

        Parameters
        ----------
        size : int | np.integer[np.typing.NBitBase] | tuple[int, ...] | None, optional
            Number or shape of random variates to generate. If None, returns a scalar.
        random_state : int | np.random.Generator | None, optional
            Seed or random number generator for reproducible sampling.

        Returns
        -------
        np.float64 | FloatArray
            Generated samples of requested shape, or empty array if sample size is zero.

        Raises
        ------
        ValueError
            If size contains a negative dimension.
        """

        target_shape: tuple[int, ...]
        if size is None:
            n_samples = 1
            target_shape = ()
        else:
            target_shape = (int(size),) if isinstance(size, (int, np.integer)) else tuple(size)
            if any(dim < 0 for dim in target_shape):
                raise ValueError(f"size must not contain negative dimensions, got {size}.")
            n_samples = int(np.prod(target_shape))
            if n_samples == 0:
                return np.empty(target_shape, dtype=np.float64)

        rng = np.random.default_rng(random_state)
        component_choices = rng.choice(self.n_components, size=n_samples, p=self.weights)
        counts = np.bincount(component_choices, minlength=self.n_components)

        samples_list = [
            np.asarray(_SAMPLING.sample(int(count), self.components[i], random_state=rng), np.float64).ravel()
            for i, count in enumerate(counts)
            if count > 0
        ]

        samples = np.concatenate(samples_list)
        rng.shuffle(samples)

        if size is None:
            return samples[0]
        return samples.reshape(target_shape)

    def __getitem__(self, key: int) -> ParametricFamilyDistribution:
        """Retrieve component by index."""

        return self.components[key]

    def __iter__(self) -> Iterator[ParametricFamilyDistribution]:
        """Iterate over mixture components."""

        return iter(self.components)

    def __copy__(self) -> "MixtureModel":
        """Return ``self``: the mixture is immutable, so a copy would be identical."""

        return self

    def __eq__(self, other: object) -> bool:
        """Check equality against another mixture, invariant to component ordering."""

        if not isinstance(other, MixtureModel):
            return NotImplemented

        if self.n_components != other.n_components:
            return False

        self_pairs = self._get_sorted_pairs()
        other_pairs = other._get_sorted_pairs()

        for (self_key, self_weight), (other_key, other_weight) in zip(self_pairs, other_pairs):
            if self_key != other_key or not np.isclose(self_weight, other_weight):
                return False

        return True

    def __hash__(self) -> int:
        """Compute a hash from components only, ensuring consistency with tolerance-based equality."""

        return hash(tuple(key for key, _ in self._get_sorted_pairs()))

    @classmethod
    def _from_normalized(
        cls,
        components: Sequence[ParametricFamilyDistribution],
        log_weights: UnivariateFloatArray,
    ) -> "MixtureModel":
        """Build a mixture from components and already normalized, read-only log-weights.

        Skips the checks and normalization of ``__init__``: both inputs must come
        from validated sources. The caches start empty.
        """

        mixture = cls.__new__(cls)
        mixture._components = tuple(components)
        mixture._cached_weights = None
        mixture._sorted_pairs_cache = None
        mixture._log_weights = log_weights
        return mixture

    def _validate_weights(self, n_components: int, weights: UnivariateFloatArray) -> None:
        if len(weights) != n_components:
            raise ValueError(f"Components number ({n_components}) must be equal to weights number ({len(weights)}).")

        if np.any(weights < 0):
            raise ValueError("Weights must be positive.")

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError(f"Sum of the weights must be equal 1, but it equal {np.sum(weights)}.")

    def _get_sorted_pairs(self) -> list[tuple[_components.ComponentKey, float]]:
        """(Component key, weight) pairs sorted so that component order does not matter."""

        if self._sorted_pairs_cache is None:
            pairs = [(_components.component_key(comp), float(w)) for comp, w in zip(self.components, self.weights)]
            self._sorted_pairs_cache = sorted(pairs)
        return self._sorted_pairs_cache
