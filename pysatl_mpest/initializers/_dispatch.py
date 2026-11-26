"""Provides dispatch of parameter estimation formulas on the family and parametrization.

Each pair ``(family name, parametrization name)`` has its own formula that estimates component
parameters from a cluster of the sample. :class:`FormulaDispatch` keeps a registry of those formulas.

The key is taken from the family, not from a component, because a component does not exist yet
when the initial mixture is being built. This makes it different from ``FamilyDispatch`` of the
M-step strategies in ``estimators``, which takes the key from a ready component, is itself callable
and has a default implementation. Registrations are made by calling the function returned by
``register`` with the implementation instead of using it as a decorator, because ``mutmut`` does not
mutate decorated functions.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable, Mapping
from types import MappingProxyType

from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName

from ..core import _components

type FormulaKey = tuple[str, str]


class FormulaDispatch[F]:
    """A registry of estimation formula implementations keyed by ``(family name, parametrization name)``.

    A family produced by ``family.view(...)`` is dispatched by its parametrization; fixed
    parameters do not affect the selection.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        register
        supports
        dispatch
    """

    def __init__(self) -> None:
        self._registry: dict[FormulaKey, F] = {}

    @property
    def registry(self) -> Mapping[FormulaKey, F]:
        """Mapping[FormulaKey, F]: Registered implementations by ``(family name, parametrization name)``."""

        return MappingProxyType(self._registry)

    def register(self, family: FamilyName | str, parametrization: str) -> Callable[[F], F]:
        """Register an implementation for a family and parametrization.

        Called as ``formulas.register(FamilyName.NORMAL, "meanStd")(implementation)``.

        Parameters
        ----------
        family : FamilyName | str
            The name of the family the implementation is for.
        parametrization : str
            The name of the parametrization of the family the implementation is for.

        Returns
        -------
        Callable
            A function that registers the implementation it is called with and returns it unchanged.

        Raises
        ------
        ValueError
            If an implementation is already registered for the family and parametrization.
        """

        key = (str(family), parametrization)

        def add(implementation: F) -> F:
            if key in self._registry:
                raise ValueError(
                    f"An implementation is already registered for family '{key[0]}' and parametrization '{key[1]}'."
                )
            self._registry[key] = implementation
            return implementation

        return add

    def supports(self, family: ParametricFamily) -> bool:
        """Check whether an implementation is registered for the family's pair of names.

        Parameters
        ----------
        family : ParametricFamily
            The family to check.

        Returns
        -------
        bool
            Whether :meth:`dispatch` would find an implementation for the family.
        """

        return _formula_key(family) in self._registry

    def dispatch(self, family: ParametricFamily) -> F:
        """Select the implementation for the family's pair of names.

        Parameters
        ----------
        family : ParametricFamily
            The family to select the implementation for.

        Returns
        -------
        F
            The implementation registered for the family and parametrization.

        Raises
        ------
        NotImplementedError
            If no implementation is registered for the family and parametrization.
        """

        name, parametrization = _formula_key(family)
        if (name, parametrization) not in self._registry:
            raise NotImplementedError(
                f"No implementation is registered for family '{name}' and parametrization '{parametrization}'."
            )
        return self._registry[name, parametrization]


def _formula_key(family: ParametricFamily) -> FormulaKey:
    """Family name and parametrization from :func:`~pysatl_mpest.core._components.family_key`."""

    name, parametrization, _ = _components.family_key(family)
    return name, parametrization
