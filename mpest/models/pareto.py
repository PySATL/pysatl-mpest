import numpy as np
from scipy.stats import pareto

from mpest.annotations import Params, Samples
from mpest.models import AModelDifferentiable, AModelWithGenerator


class Pareto(AModelWithGenerator, AModelDifferentiable):
    r"""Pareto Type I distribution model within the mpest framework.

    This class implements the Pareto Type I distribution.
    It inherits from AModelWithGenerator to provide sample generation
    functionality and from AModelDifferentiable to support derivative computation for
    parameter estimation.

    The Pareto Type I distribution is defined by two external parameters:
    - :math:`x_0 > 0`: the scale parameter (minimum value, corresponds to `scale` in scipy.stats.pareto)
    - :math:`k > 0`: the shape parameter (tail index, corresponds to `b` in scipy.stats.pareto)

    These parameters are transformed to internal model parameters:
    - :math:`\theta_{x_0} = \log(x_0)`: the log-scale parameter
    - :math:`\theta_k = \log(k)`: the log-shape parameter

    The probability density function (PDF) can be written as follows:

    External parameterization:

    .. math::

        f(x; x_0; k) = \frac{k x_0^k}{x^{k+1}} \quad \text{for} \quad x \geq x_0

    Internal parameterization:

    .. math::

        f(x; \theta_{x_0}; \theta_k) = \frac{e^{\theta_k} e^{\theta_{x_0} \cdot e^{\theta_k}}}
        {x^{e^{\theta_k} + 1}} \quad \text{for} \quad x \geq e^{\theta_{x_0}}
    """

    @property
    def name(self) -> str:
        """Return the name of the distribution.

        Returns:
            str: The string "Pareto".
        """

        return "Pareto"

    def params_convert_to_model(self, params: Params) -> Params:
        r"""Convert external parameters to internal model parameters.

        Transforms external parameters [:math:`x_0, k`] to internal parameters
        [:math:`\theta_{x_0}, \theta_k`] where :math:`\theta_{x_0} = \log(x_0)`
        and :math:`\theta_k = \log(k)`.

        Args:
            params (Params): External parameters [:math:`x_0, k`] where :math:`x_0` is the
                scale parameter and :math:`k` is the shape parameter.

        Returns:
            Params: Internal model parameters [:math:`\log(x_0), \log(k)`].
        """

        return np.log(params)

    def params_convert_from_model(self, params: Params) -> Params:
        r"""Convert internal model parameters to external parameters.

        Transforms internal model parameters [:math:`\theta_{x_0}, \theta_k`] to external
        parameters [:math:`x_0, k`] where :math:`x_0 = e^{\theta_{x_0}}` and :math:`k = e^{\theta_k}`.

        Args:
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] where
                :math:`\theta_{x_0}` is the log-scale parameter and :math:`\theta_k` is the log-shape parameter.

        Returns:
            Params: External parameters [:math:`e^{\theta_{x_0}}, e^{\theta_k}`].
        """

        return np.exp(params)

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        r"""Generate random samples from the Pareto distribution.

        Args:
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] when normalized=True,
                or external parameters [:math:`x_0, k`] when normalized=False.
            size (int, optional): Number of samples to generate. Defaults to 1.
            normalized (bool, optional): Whether the provided parameters are in normalized
                (internal) form. If False, params are treated as external parameters.
                Defaults to True.

        Returns:
            Samples: An array of randomly generated samples from the Pareto distribution.
        """

        if not normalized:
            x0, k = params
            return np.array(pareto.rvs(b=k, scale=x0, size=size))

        x0, k = self.params_convert_from_model(params)
        return np.array(pareto.rvs(b=k, scale=x0, size=size))

    def pdf(self, x: float, params: Params) -> float:
        r"""Calculate the probability density function (PDF) value at :math:`x`.

        Computes the PDF using the formula:

        .. math::

            f(x; \theta_{x_0}; \theta_k) = \frac{e^{\theta_k} e^{\theta_{x_0} \cdot e^{\theta_k}}}
            {x^{e^{\theta_k} + 1}}

        where :math:`x \geq e^{\theta_{x_0}}`.

        Args:
            x (float): The point at which to evaluate the PDF (should be at least :math:`e^{\theta_{x_0}}`).
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] where
                :math:`\theta_{x_0}` is the log-scale parameter and :math:`\theta_k` is the log-shape parameter.

        Returns:
            float: The PDF value at point :math:`x`, or 0 if :math:`x < e^{\theta_{x_0}}`.
        """

        theta_x0, theta_k = params
        e_x0 = np.exp(theta_x0)
        if x < e_x0:
            return 0.0

        e_k = np.exp(theta_k)
        num = e_k * np.exp(theta_x0 * e_k)
        denom = x ** (e_k + 1)

        return num / denom

    def lpdf(self, x: float, params: Params) -> float:
        r"""Calculate the natural logarithm of the probability density function at :math:`x`.

        Computes the log-PDF using the formula:

        .. math::

            \ln(f(x; \theta_{x_0}; \theta_k)) = \theta_k + e^{\theta_k} (\theta_{x_0} - \ln(x)) - \ln(x)

        where :math:`x \geq e^{\theta_{x_0}}`.

        Args:
            x (float): The point at which to evaluate the log-PDF (should be at least :math:`e^{\theta_{x_0}}`).
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] where
                :math:`\theta_{x_0}` is the log-scale parameter and :math:`\theta_k` is the log-shape parameter.

        Returns:
            float: The logarithm of the PDF value at point :math:`x`, or :math:`-\infty` if
            :math:`x < e^{\theta_{x_0}}`.
        """

        theta_x0, theta_k = params
        e_x0, e_k = np.exp(params)
        log = np.log(x)

        if x < e_x0:
            return -np.inf

        return theta_k + e_k * (theta_x0 - log) - log

    def ld_theta_x0(self, x: float, params: Params) -> float:
        r"""Calculate the partial derivative of the log-PDF with respect to :math:`\theta_{x_0}`.

        Computes the derivative:

        .. math::

            \frac{\partial\ln(f(x; \theta_{x_0}; \theta_k))}{\partial \theta_{x_0}} = e^{\theta_k}

        where :math:`x \geq e^{\theta_{x_0}}`.

        Args:
            x (float): The point at which to evaluate the derivative (should be at least :math:`e^{\theta_{x_0}}`).
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] where
                :math:`\theta_{x_0}` is the log-scale parameter and :math:`\theta_k` is the log-shape parameter.

        Returns:
            float: The partial derivative of the log-PDF with respect to :math:`\theta_{x_0}` at
            point :math:`x`, or :math:`-\infty` if :math:`x < e^{\theta_{x_0}}`.
        """

        e_x0, e_k = np.exp(params)

        if x < e_x0:
            return -np.inf

        return e_k

    def ld_theta_k(self, x: float, params: Params) -> float:
        r"""Calculate the partial derivative of the log-PDF with respect to :math:`\theta_k`.

        Computes the derivative:

        .. math::

            \frac{\partial\ln(f(x; \theta_{x_0}; \theta_k))}{\partial \theta_k} =
            1 + e^{\theta_k} (\theta_{x_0} - \ln(x))

        where :math:`x \geq e^{\theta_{x_0}}`.

        Args:
            x (float): The point at which to evaluate the derivative (should be at least :math:`e^{\theta_{x_0}}`).
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] where
                :math:`\theta_{x_0}` is the log-scale parameter and :math:`\theta_k` is the log-shape parameter.

        Returns:
            float: The partial derivative of the log-PDF with respect to :math:`\theta_k` at
            point :math:`x`, or :math:`-\infty` if :math:`x < e^{\theta_{x_0}}`.
        """

        theta_x0, theta_k = params
        e_x0 = np.exp(params[0])

        if x < e_x0:
            return -np.inf

        return 1 + np.exp(theta_k) * (theta_x0 - np.log(x))

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        r"""Calculate the gradient of the log-PDF with respect to all parameters.

        Args:
            x (float): The point at which to evaluate the gradient (should be at least :math:`e^{\theta_{x_0}}`).
            params (Params): Internal model parameters [:math:`\theta_{x_0}, \theta_k`] where
                :math:`\theta_{x_0}` is the log-scale parameter and :math:`\theta_k` is the log-shape parameter.

        Returns:
            np.ndarray: An array containing the partial derivatives of the log-PDF with
            respect to each parameter [:math:`\theta_{x_0}, \theta_k`] at point :math:`x`.
        """

        return np.array([self.ld_theta_x0(x, params), self.ld_theta_k(x, params)])
