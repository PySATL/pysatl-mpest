"""Provides dispatch of M-step strategies on the family and parametrization of a component.

Components are `pysatl-core` distributions of one class, so `functools.singledispatch`, which selects
an implementation by the class of the first argument, cannot tell a normal distribution from an
exponential one. `FamilyDispatch` selects it by the pair ``(family name, parametrization name)``
instead, and falls back to a default implementation when nothing is registered for the pair.

A `FamilyDispatch` is not a decorator: it is constructed with the default implementation, an explicit
``name`` and a ``doc`` describing the whole dispatch, and implementations are added through its
``register`` method.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Concatenate

from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName

type StrategyKey = tuple[str, str]


class FamilyDispatch[**P, R]:
    """A function whose implementation is selected by the family and parametrization of its first argument.

    The first argument must be a component of the mixture. A component created from ``family.view(...)``
    belongs to the same family as the one it was built from, so it is dispatched like a plain one.

    Parameters
    ----------
    default : Callable[Concatenate[ParametricFamilyDistribution, P], R]
        The implementation used when nothing is registered for the family and parametrization
        of the component.
    name : str
        The dispatcher's own ``__name__`` and ``__qualname__``.
    doc : str
        The dispatcher's own docstring, describing the whole dispatch and shown by ``help()``.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        register
        dispatch
    """

    __name__: str
    __qualname__: str
    __doc__: str

    def __init__(
        self,
        default: Callable[Concatenate[ParametricFamilyDistribution, P], R],
        *,
        name: str,
        doc: str,
    ) -> None:
        self._default = default
        self._registry: dict[StrategyKey, Callable[Concatenate[ParametricFamilyDistribution, P], R]] = {}
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = doc

    @property
    def registry(self) -> Mapping[StrategyKey, Callable[Concatenate[ParametricFamilyDistribution, P], R]]:
        """Mapping[StrategyKey, Callable]: Registered implementations by ``(family name, parametrization name)``."""

        return MappingProxyType(self._registry)

    def register(
        self, family: FamilyName | str, parametrization: str
    ) -> Callable[
        [Callable[Concatenate[ParametricFamilyDistribution, P], R]],
        Callable[Concatenate[ParametricFamilyDistribution, P], R],
    ]:
        """Register an implementation for a family and parametrization. Meant to be used as a decorator.

        Parameters
        ----------
        family : FamilyName | str
            The name of the family the implementation is for.
        parametrization : str
            The name of the parametrization of the family the implementation is for.

        Returns
        -------
        Callable
            A decorator that registers the function and returns it unchanged.

        Raises
        ------
        ValueError
            If an implementation is already registered for the family and parametrization.
        """

        key = (str(family), parametrization)

        def decorator(
            func: Callable[Concatenate[ParametricFamilyDistribution, P], R],
        ) -> Callable[Concatenate[ParametricFamilyDistribution, P], R]:
            if key in self._registry:
                raise ValueError(
                    f"An implementation is already registered for family '{key[0]}' and parametrization '{key[1]}'."
                )
            self._registry[key] = func
            return func

        return decorator

    def dispatch(
        self, component: ParametricFamilyDistribution
    ) -> Callable[Concatenate[ParametricFamilyDistribution, P], R]:
        """Select the implementation for a component.

        Parameters
        ----------
        component : ParametricFamilyDistribution
            The component to select the implementation for.

        Returns
        -------
        Callable
            The implementation registered for the family and parametrization of the component,
            or the default one.
        """

        return self._registry.get((component.family_name, component.parametrization_name), self._default)

    def __call__(self, component: ParametricFamilyDistribution, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.dispatch(component)(component, *args, **kwargs)
