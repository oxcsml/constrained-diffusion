import jax
import jax.numpy as jnp

from score_sde.sde import SDE, RSDE as RSDEBase, get_matrix_div_fn
from score_sde.models import get_score_fn
from score_sde.utils import batch_mul
from riemannian_score_sde.sampling import get_discrete_pc_sampler
from riemannian_score_sde.models.distribution import (
    UniformDistribution,
    MultivariateNormal,
    WrapNormDistribution,
)


class RSDE(RSDEBase):
    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde, score_fn)
        self.manifold = sde.manifold


class Langevin(SDE):
    """Construct Langevin dynamics on a manifold"""

    def __init__(
        self,
        beta_schedule,
        manifold,
        ref_scale=0.5,
        ref_mean=None,
        N=100,
    ):
        super().__init__(beta_schedule)
        self.manifold = manifold
        self.limiting = WrapNormDistribution(manifold, scale=ref_scale, mean=ref_mean)
        self.N = N

    def drift(self, x, t, score=None):
        """dX_t =-0.5 beta(t) grad U(X_t)dt + sqrt(beta(t)) dB_t"""

        def fixed_grad(grad):
            is_nan_or_inf = jnp.isnan(grad) | (jnp.abs(grad) == jnp.inf)
            return jnp.where(is_nan_or_inf, jnp.zeros_like(grad), grad)

        drift_fn = jax.vmap(lambda x: -0.5 * fixed_grad(self.limiting.grad_U(x)))
        beta_t = self.beta_schedule.beta_t(t)
        drift = beta_t[..., None] * drift_fn(x)
        return drift

    def marginal_sample(self, rng, x, N, return_hist=False):
        sampler = get_discrete_pc_sampler(
            self, predictor="GRW", return_hist=return_hist
        )
        out, t = sampler(rng, x, N=N)
        return out, t

    def marginal_prob(self, x, t):
        # NOTE: this is only a proxy!
        log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
        axis_to_expand = tuple(range(-1, -len(x.shape), -1))  # (-1) or (-1, -2)
        mean_coeff = jnp.expand_dims(jnp.exp(log_mean_coeff), axis=axis_to_expand)
        # mean = jnp.exp(log_mean_coeff)[..., None] * x
        mean = mean_coeff * x
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def varhadan_exp(self, xs, xt, s, t):
        delta_t = self.beta_schedule.rescale_t(t) - self.beta_schedule.rescale_t(s)
        axis_to_expand = tuple(range(-1, -len(xt.shape), -1))  # (-1) or (-1, -2)
        delta_t = jnp.expand_dims(delta_t, axis=axis_to_expand)
        grad = self.manifold.log(xs, xt) / delta_t
        return delta_t, grad

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class VPSDE(Langevin):
    def __init__(self, beta_schedule, manifold=None, **kwargs):
        super().__init__(beta_schedule, manifold)
        self.limiting = MultivariateNormal(dim=manifold.dim)

    def marginal_sample(self, rng, x, t):
        mean, std = self.marginal_prob(x, t)
        z = jax.random.normal(rng, x.shape)
        return mean + batch_mul(std, z)

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        mean, std = self.marginal_prob(x0, t)
        std = jnp.expand_dims(std, -1)
        score = -1 / (std**2) * (x - mean)
        logp = None
        return logp, score


class Brownian(Langevin):
    def __init__(
        self,
        manifold,
        beta_schedule,
        N=100,
        boundary_enforce=False,
        boundary_dis=0.01,
        std_trick=True,
    ):
        """Construct a Brownian motion on a compact manifold"""
        super().__init__(beta_schedule, manifold, N=N)
        # self.manifold = manifold
        self.limiting = UniformDistribution(manifold)
        self.boundary_enforce = boundary_enforce
        self.boundary_dis = boundary_dis
        self.std_trick = std_trick

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        s = self.beta_schedule.rescale_t(t)
        logp_grad = self.manifold.grad_marginal_log_prob(x0, x, s, **kwargs)
        return None, logp_grad

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(
            self,
            score_fn,
            *args,
            std_trick=self.std_trick,
            residual_trick=False,
            boundary_enforce=self.boundary_enforce,
            boundary_dis=self.boundary_dis
        )


class ReflectedBrownian(Brownian):
    def marginal_sample(self, rng, x, N, return_hist=False):
        t = N / self.N
        noise = self.beta_schedule.rescale_t(t) ** 0.5 * jax.random.normal(
            rng, x.shape, x.dtype
        )
        return self.manifold.metric.exp(noise, x), t


class HessianSDE(Langevin):
    def __init__(
        self,
        manifold,
        beta_schedule,
        N=100,
        boundary_enforce=False,
        boundary_dis=0.001,
        full_diffusion_matrix=True,
        std_trick=True,
    ):
        super().__init__(beta_schedule, manifold, N=N)
        self.limiting = UniformDistribution(manifold)
        self.full_diffusion_matrix = full_diffusion_matrix
        self.rsde_reverse_drift = True
        self.boundary_enforce = boundary_enforce
        self.boundary_dis = boundary_dis
        self.std_trick = std_trick

    def drift(self, x, t, score=None):
        beta_t = self.beta_schedule.beta_t(t)
        G_inv = self.manifold.metric.metric_inverse_matrix

        # assume the full diffusion mat is coming from metric being diagonal or not
        if hasattr(self.manifold.metric, "div_metric_inverse_matrix"):
            div_term = self.manifold.metric.div_metric_inverse_matrix(x)
        else:
            div_term = get_matrix_div_fn(G_inv)(x, t, None)

        drift = 1 / 2 * beta_t[..., None] * div_term
        return drift
        return jnp.zeros_like(x)

    def diffusion(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        if self.full_diffusion_matrix:
            diffusion = jnp.sqrt(beta_t)[
                ..., None, None
            ] * self.manifold.metric.metric_inverse_matrix_sqrt(x)
        else:
            diffusion = jnp.sqrt(beta_t)[
                ..., None
            ] * self.manifold.metric.metric_inverse_matrix_sqrt(x)
        return diffusion

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(
            self,
            score_fn,
            *args,
            std_trick=self.std_trick,
            residual_trick=False,
            boundary_enforce=self.boundary_enforce,
            boundary_dis=self.boundary_dis
        )
