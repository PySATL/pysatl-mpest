import numpy as np
from scipy.special import betaln, digamma, expm1
from scipy.stats import beta

from mpest.annotations import Params, Samples
from mpest.models import AModelDifferentiable, AModelWithGenerator


class Beta(AModelWithGenerator, AModelDifferentiable):
    r"""Beta probability distribution model.

    This class implements the Beta distribution, a continuous probability distribution
    defined on the interval [0, 1].

    It inherits functionality for sample generation from AModelWithGenerator
    and for computing derivatives from AModelDifferentiable.

    The Beta distribution is parameterized by two positive shape parameters:

    - External parameters: :math:`a` and :math:`b` (:math:`a,b > 0`)
    - Internal (model) parameters: :math:`\mu = \log(a)` and :math:`\nu = \log(b)`

    The probability density function (PDF) with external parameters is:

    .. math::

        f(x; a, b) = \frac{x^{a-1} (1-x)^{b-1}}{B(a, b)}

    where :math:`B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}` is the Beta function,
    and :math:`0 < x < 1`.

    The PDF with internal parameters is:

    .. math::

        f(x; \mu, \nu) = \exp\left( (e^\\mu - 1)\log(x) + (e^\nu - 1)\log(1-x)
        - \mathrm{betaln}(e^\mu, e^\nu) \right)
    """

    @property
    def name(self) -> str:
        """Returns the name of the distribution.

        Returns:
            str: The name of the distribution "Beta".
        """

        return "Beta"

    def params_convert_to_model(self, params: Params) -> Params:
        """Converts external parameters to internal model parameters.

        Transforms the external parameters [:math:`a, b`] to internal model parameters
        [:math:`\\mu, \\nu`] where :math:`\\mu = \\log(a)` and :math:`\\nu = \\log(b)`.

        Args:
            params (Params): External parameters [:math:`a, b`] where :math:`a` and :math:`b`
                are the shape parameters of the Beta distribution (:math:`a,b > 0`).

        Returns:
            Params: Internal model parameters [:math:`\\log(a), \\log(b)`].
        """

        return np.log(params)

    def params_convert_from_model(self, params: Params) -> Params:
        r"""Converts internal model parameters to external parameters.

        Transforms the internal model parameters [:math:`\mu, \nu`] to external
        parameters [:math:`a, b`] where :math:`a = e^{\mu}` and :math:`b = e^{\nu}`.

        Args:
            params (Params): Internal model parameters [:math:`\mu, \nu`] where
                :math:`\mu` and :math:`\nu` are the log-shape parameters.

        Returns:
            Params: External parameters [:math:`e^{\mu}, e^{\nu}`].
        """

        return np.exp(params)

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        r"""Generates random samples from the Beta distribution.

        Args:
            params (Params): Internal model parameters [:math:`\mu, \nu`] when normalized=True,
                or external parameters [:math:`a, b`] when normalized=False.
            size (int, optional): Number of samples to generate. Defaults to 1.
            normalized (bool, optional): Whether the provided parameters are in normalized
                (internal) form. If False, params are treated as external parameters.
                Defaults to True.

        Returns:
            Samples: An array of randomly generated samples from the Beta distribution.
        """

        if not normalized:
            return np.array(beta.rvs(a=params[0], b=params[1], size=size))

        a, b = self.params_convert_from_model(params)
        return np.array(beta.rvs(a=a, b=b, size=size))

    def pdf(self, x: float, params: Params) -> float:
        """Calculates the probability density function (PDF) value at :math:`x`.

        The PDF value is computed by exponentiating the result of the log-PDF function.

        Args:
            x (float): The point at which to evaluate the PDF (should be in (0, 1)).
            params (Params): Internal model parameters [:math:`\\mu, \\nu`] where :math:`\\mu`
                and :math:`\\nu` are the log-shape parameters.

        Returns:
            float: The PDF value at point :math:`x`.
        """

        return np.exp(self.lpdf(x, params))

    def lpdf(self, x: float, params: Params) -> float:
        r"""Calculates the natural logarithm of the probability density function at :math:`x`.

        Computes the log-PDF using the formula:

        .. math::

            \ln(f(x; \mu, \nu)) = (e^\mu - 1)\log(x) + (e^\nu - 1)\log(1-x)
            - \mathrm{betaln}(e^\mu, e^\nu)

        where :math:`0 < x < 1`.

        Args:
            x (float): The point at which to evaluate the log-PDF (should be in (0, 1)).
            params (Params): Internal model parameters [:math:`\mu, \nu`] where :math:`\mu`
                and :math:`\nu` are the log-shape parameters.

        Returns:
            float: The logarithm of the PDF value at point :math:`x`, or :math:`-\infty` if
            :math:`x` is not in (0, 1).
        """

        if not (0 < x < 1):
            return -np.inf

        mu, nu = params
        lbeta = -betaln(np.exp(params[0]), np.exp(params[1]))
        log1 = expm1(mu) * np.log(x)
        log2 = expm1(nu) * np.log(1 - x)

        return lbeta + log1 + log2

    def ldmu(self, x: float, params: Params) -> float:
        r"""Calculates the partial derivative of the log-PDF with respect to :math:`\mu`.

        Computes the derivative:

        .. math::

            \frac{\partial\ln(f(x; \mu, \nu))}{\partial \mu} =
            e^\mu (\psi(e^\mu + e^\nu) - \psi(e^\mu) + \log(x))

        where :math:`\psi` is the digamma function and :math:`0 < x < 1`.

        Args:
            x (float): The point at which to evaluate the derivative (should be in (0, 1)).
            params (Params): Internal model parameters [:math:`\mu, \nu`] where :math:`\mu`
                and :math:`\nu` are the log-shape parameters.

        Returns:
            float: The partial derivative of the log-PDF with respect to :math:`\mu` at
            point :math:`x`, or :math:`-\infty` if :math:`x` is not in (0, 1).
        """

        if not (0 < x < 1):
            return -np.inf

        a, b = self.params_convert_from_model(params)
        return a * (digamma(a + b) - digamma(a) + np.log(x))

    def ldnu(self, x: float, params: Params) -> float:
        r"""Calculates the partial derivative of the log-PDF with respect to :math:`\nu`.

        Computes the derivative:

        .. math::

            \frac{\partial\ln(f(x; \mu, \nu))}{\partial \nu} =
            e^\nu (\psi(e^\mu + e^\nu) - \psi(e^\nu) + \log(1-x))

        where :math:`\psi` is the digamma function and :math:`0 < x < 1`.

        Args:
            x (float): The point at which to evaluate the derivative (should be in (0, 1)).
            params (Params): Internal model parameters [:math:`\mu, \nu`] where :math:`\mu`
                and :math:`\nu` are the log-shape parameters.

        Returns:
            float: The partial derivative of the log-PDF with respect to :math:`\nu` at
            point :math:`x`, or -infinity if :math:`x` is not in (0, 1).
        """

        if not (0 < x < 1):
            return -np.inf

        a, b = self.params_convert_from_model(params)
        return b * (digamma(a + b) - digamma(b) + np.log(1 - x))

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        r"""Calculates the gradient of the log-PDF with respect to all parameters.

        Args:
            x (float): The point at which to evaluate the gradient (should be in (0, 1)).
            params (Params): Internal model parameters [:math:`\mu, \nu`] where :math:`\mu`
                and :math:`\nu` are the log-shape parameters.

        Returns:
            np.ndarray: An array containing the partial derivatives of the log-PDF with
            respect to each parameter [:math:`\mu, \nu`] at point :math:`x`.
        """

        return np.array([self.ldmu(x, params), self.ldnu(x, params)])
