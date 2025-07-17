import numpy as np
from scipy.stats import cauchy

from mpest.annotations import Params, Samples
from mpest.models import AModelDifferentiable, AModelWithGenerator


class Cauchy(AModelWithGenerator, AModelDifferentiable):
    """Cauchy distribution model implementation in the mpest framework.

    This class implements the Cauchy probability distribution.
    It inherits functionality for sample generation from AModelWithGenerator
    and for computing derivatives from AModelDifferentiable.

    The Cauchy distribution is parameterized by:
      - External parameters: location (:math:`x_0`) and scale (:math:`\\gamma > 0`)
      - Internal model parameters: location (:math:`x_0`) and log-scale (:math:`sigma = \\log(\\gamma)`)

    The probability density function is:

    .. math::

        `f(x; x_0, \\gamma) = \frac{1}{\\pi} \frac{\\gamma}{(x - x_0)^2 + \\gamma^2}`

    In terms of internal parameters:
    .. math::

        `f(x; x_0, \\sigma) = \frac{1}{\\pi} \frac{e^{-\\sigma}}{1+((x-x_0)e^{-\\sigma})^2}`

    """

    @property
    def name(self) -> str:
        """Returns the name of the distribution.

        Returns:
            str: The name of the distribution ('Cauchy').
        """

        return "Cauchy"

    def params_convert_to_model(self, params: Params) -> Params:
        r"""Converts external parameters to internal model parameters.

        Transforms the external parameterization [:math:`x_0, \gamma`] to internal model
        parameters [:math:`x_0, \sigma`] where :math:`\sigma = \log(\gamma)`.

        Args:
            params (Params): External parameters [:math:`x_0, \gamma`] where :math:`x_0` is the location
                parameter and gamma is the scale parameter.

        Returns:
            Params: Internal model parameters [:math:`x_0, \log(\gamma)`].
        """

        return np.array([params[0], np.log(params[1])])

    def params_convert_from_model(self, params: Params) -> Params:
        r"""Converts internal model parameters to external parameters.

        Transforms the internal model parameters [:math:`x_0, \sigma`] to external
        parameterization [:math:`x_0, \gamma`] where :math:`\gamma = e^{\sigma}`.

        Args:
            params (Params): Internal model parameters [:math:`x_0, \sigma`] where :math:`x_0` is the location
                parameter and :math:`\sigma` is the log-scale parameter.

        Returns:
            Params: External parameters [:math:`x_0, e^{\sigma}`].
        """

        return np.array([params[0], np.exp(params[1])])

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        r"""Generates random samples from the Cauchy distribution.

        Args:
            params (Params): Internal model parameters [:math:`x_0, \sigma`] when normalized=True,
                or external parameters [:math:`x_0, \gamma`] when normalized=False.
            size (int, optional): Number of samples to generate. Defaults to 1.
            normalized (bool, optional): Whether the provided parameters are in normalized
                (internal) form. If False, params are treated as external parameters.
                Defaults to True.

        Returns:
            Samples: An array of randomly generated samples from the Cauchy distribution.
        """

        if not normalized:
            x0, gamma = params
            return np.array(cauchy.rvs(loc=x0, scale=gamma, size=size))

        x0, gamma = self.params_convert_from_model(params)
        return np.array(cauchy.rvs(loc=x0, scale=gamma, size=size))

    def pdf(self, x: float, params: Params) -> float:
        """Calculates the probability density function (PDF) value at :math:`x`.

        Computes the PDF value using the formula:

        .. math::

            `f(x; x_0; \\sigma) = \\frac{1}{\\pi} \\frac{e^{-\\sigma}}{1+((x-x_0)e^{-\\sigma})^2}`

        Args:
            x (float): The point at which to evaluate the PDF.
            params (Params): Internal model parameters [:math:`x_0, \\sigma`] where :math:`x_0` is the location
                parameter and :math:`sigma` is the log-scale parameter.

        Returns:
            float: The PDF value at point :math:`x`.
        """

        x0, sigma = params
        e_neg = np.exp(-sigma)
        t = (x - x0) * e_neg

        return (1 / np.pi) * e_neg / (1 + t * t)

    def lpdf(self, x: float, params: Params) -> float:
        """Calculates the natural logarithm of the probability density function at :math:`x`.

        Computes the log-PDF using the formula:

        .. math::

            `\\ln(f(x; x0; \\sigma)) = -\\ln(\\pi) - \\sigma
            - \\ln\\Big(1 + \\left(\\frac{x - x_0}{e^{\\sigma}}\\right)^2\\Big)`

        Args:
            x (float): The point at which to evaluate the log-PDF.
            params (Params): Internal model parameters [:math:`x_0, \\sigma`] where :math:`x_0` is the location
                parameter and :math:`\\sigma` is the log-scale parameter.

        Returns:
            float: The logarithm of the PDF value at point :math:`x`.
        """

        x0, sigma = params
        gamma = np.exp(sigma)
        z = (x - x0) / gamma
        return -np.log(np.pi) - sigma - np.log(1 + z * z)

    def ldx0(self, x: float, params: Params) -> float:
        """Calculates the partial derivative of the log-PDF with respect to :math:`x_0`.

        Computes the derivative:

        .. math::

            `\\frac{\\partial\\ln(f(x; x_0; \\sigma))}{\\partial x_0} = \\frac{2z}{(z^2 + 1) \\cdot e^{\\sigma}}$$

        where :math:`z = \\dfrac{x - x_0}{e^{\\sigma}}`

        Args:
            x (float): The point at which to evaluate the derivative.
            params (Params): Internal model parameters [:math:`x_0, \\sigma`] where :math:`x_0` is the location
                parameter and :math:`\\sigma` is the log-scale parameter.

        Returns:
            float: The partial derivative of the log-PDF with respect to :math:`x_0` at point :math:`x`.
        """

        x0, sigma = params
        gamma = np.exp(sigma)
        z = (x - x0) / gamma
        z2 = z * z
        return 2 * z / (z2 + 1) / gamma

    def ldsigma(self, x: float, params: Params) -> float:
        """Calculates the partial derivative of the log-PDF with respect to :math:`\\sigma`.

        Computes the derivative:

        .. math::

            `\\frac{\\partial\\ln(f(x; x_0; \\sigma))}{\\partial \\sigma} = \\frac{z^2 - 1}{z^2 + 1}`

        where :math:`z = \\dfrac{x - x_0}{e^{\\sigma}}`

        Args:
            x (float): The point at which to evaluate the derivative.
            params (Params): Internal model parameters [x0, sigma] where x0 is the location
                parameter and sigma is the log-scale parameter.

        Returns:
            float: The partial derivative of the log-PDF with respect to sigma at point x.
        """

        x0, sigma = params
        gamma = np.exp(sigma)
        z = (x - x0) / gamma
        z2 = z * z
        return (z2 - 1) / (z2 + 1)

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        r"""Calculates the gradient of the log-PDF with respect to all parameters.

        Args:
            x (float): The point at which to evaluate the gradient.
            params (Params): Internal model parameters [:math:`x_0, \sigma`] where :math:`x_0` is the location
                parameter and :math:`sigma` is the log-scale parameter.

        Returns:
            np.ndarray: An array containing the partial derivatives of the log-PDF with
            respect to each parameter at point :math:`x`.
        """

        return np.array([self.ldx0(x, params), self.ldsigma(x, params)])
