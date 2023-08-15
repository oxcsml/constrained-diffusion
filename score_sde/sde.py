"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Modified code from https://github.com/yang-song/score_sde
"""
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from score_sde.models.model import get_score_fn
from score_sde.utils import get_exact_div_fn
from score_sde.schedule import ConstantBetaSchedule


class SDE(ABC):
    # Specify if the sde returns full diffusion matrix, or just a scalar indicating diagonal variance

    full_diffusion_matrix = False
    rsde_reverse_drift = False

    def __init__(self, beta_schedule=ConstantBetaSchedule(), N=100):
        """Abstract definition of an SDE"""
        self.beta_schedule = beta_schedule
        self.tf = beta_schedule.tf
        self.t0 = beta_schedule.t0
        self.lambda0 = beta_schedule.lambda0
        self.psi = beta_schedule.psi
        self.N = N

    @abstractmethod
    def drift(self, x, t, score=None):
        """Compute the drift coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        pass

    def diffusion(self, x, t):
        """Compute the diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        beta_t = self.beta_schedule.beta_t(t)
        return jnp.sqrt(beta_t)

    def coefficients(self, x, t, score=None):
        """Compute the drift and diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        return self.drift(x, t, score=score), self.diffusion(x, t)

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        raise NotImplementedError()

    def marginal_log_prob(self, x0, x, t):
        """Compute the log marginal distribution of the SDE, $log p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x0: jnp.ndarray
            Location of the start of the diffusion
        x : jnp.ndarray
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        raise NotImplementedError()

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        """Compute the log marginal distribution and its gradient

        Parameters
        ----------
        x0: jnp.ndarray
            Location of the start of the diffusion
        x : jnp.ndarray
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """

        def marginal_log_prob(x0, x, t):
            return self.marginal_log_prob(x0, x, t, **kwargs)

        logp_grad_fn = jax.value_and_grad(marginal_log_prob, argnums=1, has_aux=False)
        logp, logp_grad = jax.vmap(logp_grad_fn)(x0, x, t)
        logp_grad = self.manifold.to_tangent(logp_grad, x)
        return logp, logp_grad

    def sample_limiting_distribution(self, rng, shape):
        """Generate samples from the limiting distribution, $p_{t_f}(x)$.
        (distribution may not exist / be inexact)

        Parameters
        ----------
        rng : jnp.random.KeyArray
        shape : Tuple
            Shape of the samples to sample.
        """
        return self.limiting.sample(rng, shape)

    def limiting_distribution_logp(self, z):
        """Compute log-density of the limiting distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: limiting distribution sample
        Returns:
          log probability density
        """
        return self.limiting.log_prob(z)

    def discretize(self, x, t, dt):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at

        Returns:
            f, G - the discretised SDE drift and diffusion coefficients
        """
        drift, diffusion = self.coefficients(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(jnp.abs(dt))
        return f, G

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(self, score_fn, *args)

    def reverse(self, score_fn):
        return RSDE(self, score_fn)

    def probability_ode(self, score_fn):
        return ProbabilityFlowODE(self, score_fn)


class ProbabilityFlowODE:
    def __init__(self, sde: SDE, score_fn=None):
        self.sde = sde

        self.t0 = sde.t0
        self.tf = sde.tf
        self.N = sde.N

        if score_fn is None and not isinstance(sde, RSDE):
            raise ValueError(
                "Score function must be not None or SDE must be a reversed SDE"
            )
        elif score_fn is not None:
            self.score_fn = score_fn
        elif isinstance(sde, RSDE):
            self.score_fn = sde.score_fn

    def coefficients(self, x, t, z=None, score=None):
        drift, diffusion = self.sde.coefficients(x, t, score=score)
        score_fn = self.score_fn(x, t, z)
        # compute G G^T score_fn
        if self.sde.full_diffusion_matrix:
            # if square matrix diffusion coeffs
            ode_drift = drift - 0.5 * jnp.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            ode_drift = drift - 0.5 * jnp.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        return ode_drift, jnp.zeros(drift.shape[:-1])


def get_matrix_div_fn(func):
    def matrix_div_fn(x, t, context):
        # define function that returns div of nth column matrix function
        def f(n):
            return get_exact_div_fn(lambda x, t, context: func(x)[..., n])(
                x, t, context
            )

        matrix = func(x)
        div_term = jax.vmap(f)(jnp.arange(matrix.shape[-1]))
        div_term = jnp.moveaxis(div_term, 0, -1)
        return div_term

    return matrix_div_fn


class RSDE(SDE):
    """Reverse time SDE, assuming the diffusion coefficient is spatially homogenous"""

    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde.beta_schedule.reverse(), N=sde.N)
        self.sde = sde
        self.score_fn = score_fn

    def diffusion(self, x, t):
        temperature_mul_diff = jnp.sqrt(1 + self.sde.psi)
        return temperature_mul_diff * self.sde.diffusion(x, t)

    def drift(self, x, t, score=None):
        forward_drift, diffusion = self.sde.coefficients(x, t)
        if score is None:
            score = self.score_fn(x, t)
        temperature_mul_score = (self.sde.psi / 2 + 1) * self.sde.lambda0
        score *= temperature_mul_score

        if self.sde.rsde_reverse_drift:
            forward_drift *= -1

        # compute G G^T score_fn
        if self.sde.full_diffusion_matrix:
            # if square matrix diffusion coeffs
            reverse_drift = forward_drift - jnp.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score
            )
        elif (
            len(diffusion.shape) > 1 and diffusion.shape[-1] == forward_drift.shape[-1]
        ):
            # Diagonal diffusion
            reverse_drift = forward_drift - jnp.einsum(
                "...i,...i,...i->...i", diffusion, diffusion, score
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            reverse_drift = forward_drift - jnp.einsum(
                "...,...,...i->...i", diffusion, diffusion, score
            )

        return reverse_drift

    def reverse(self):
        return self.sde


class VPSDE(SDE):
    def __init__(self, beta_schedule):
        from score_sde.models.distribution import NormalDistribution

        super().__init__(beta_schedule)
        self.limiting = NormalDistribution()

    def drift(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        return -0.5 * beta_t[..., None] * x

    def marginal_prob(self, x, t):
        log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
        mean = jnp.exp(log_mean_coeff)[..., None] * x
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std


class VESDE(SDE):
    pass
