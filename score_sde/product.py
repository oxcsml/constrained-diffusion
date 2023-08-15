import jax
import jax.numpy as jnp
import jax.random as random

from score_sde.models.model import get_score_fn
from score_sde.sampling import Predictor, Corrector, get_predictor, get_corrector
from score_sde.sde import ProbabilityFlowODE

class ProductSDE:
    full_diffusion_matrix = False

    def __init__(self, sdes, tf=1.0, t0=0, N=100, boundary_enforce=False, boundary_dis=0.1, std_trick=False):
        self.tf, self.t0, self.N = tf, t0, N
        self.sdes = sdes
        self.boundary_enforce = boundary_enforce
        self.boundary_dis = boundary_dis
        self.std_trick = std_trick

        for sde in self.sdes:
            sde.beta_schedule.tf = self.tf
            sde.beta_schedule.t0 = self.t0
            sde.N = self.N

    def sample_limiting_distribution(self, rng, shape):
        keys = jax.random.split(rng, len(self.sdes))
        return [
            self.sdes[i].sample_limiting_distribution(keys[i], shape)
            for i in range(len(self.sdes))
        ]

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(
            self, score_fn, *args, std_trick=self.std_trick, residual_trick=False,
            boundary_enforce=self.boundary_enforce, boundary_dis=self.boundary_dis
        )

    def marginal_sample(self, rng, x, N, return_hist=False):
        sampler = get_product_discrete_pc_sampler(
            self, predictor="GRW", return_hist=return_hist
        )
        out, t = sampler(rng, x, N=N)
        return out, t
    
    def probability_ode(self, score_fn):
        return ProductProbabilityFlowODE(self, score_fn)


class ProductRSDE(ProductSDE):
    def __init__(self, product, score_fn):
        self.score_fn = score_fn
        sdes = [sde.reverse(score_fn) for sde in product.sdes]
        super().__init__(sdes, tf=product.t0, t0=product.tf, N=product.N)
        
        
class ProductProbabilityFlowODE:
    def __init__(self, product, score_fn=None):
        self.product = product

        self.t0 = product.t0
        self.tf = product.tf
        self.N = product.N

        if score_fn is None and not isinstance(sde, RSDE):
            raise ValueError(
                "Score function must be not None or SDE must be a reversed SDE"
            )
        elif score_fn is not None:
            self.score_fn = score_fn
        elif isinstance(sde, RSDE):
            self.score_fn = sde.score_fn

    def coefficients(self, xstack, t, z=None, base_scores=None):
        xs = [xstack[..., :3], xstack[..., 3:]] if not isinstance(xstack, list) else xstack
        scores = self.score_fn(xs, t, z)
        
        ode_drifts = [None] * len(xs)
        
        for i, (sde, x, score) in enumerate(zip(self.product.sdes, xs, scores)):
            drift, diffusion = sde.coefficients(x, t, score=base_scores)

            # compute G G^T score_fn
            if sde.full_diffusion_matrix:
                # if square matrix diffusion coeffs
                ode_drift = drift - 0.5 * jnp.einsum(
                    "...ij,...kj,...k->...i", diffusion, diffusion, score
                )
            else:
                # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
                ode_drift = drift - 0.5 * jnp.einsum(
                    "...,...,...i->...i", diffusion, diffusion, score
                )
                
            ode_drifts[i] = ode_drift
            
        ode_drift = jnp.concatenate(ode_drifts, axis=-1)
        return ode_drift, jnp.zeros(ode_drift.shape[:-1])


def get_product_discrete_pc_sampler(
    product: ProductSDE,
    predictor: Predictor = "EulerMaruyamaPredictor",
    corrector: Corrector = None,
    inverse_scaler=lambda x: x,
    snr: float = 0.2,
    n_steps: int = 1,
    denoise: bool = True,
    eps: float = 1e-3,
    return_hist: bool = False,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      N: An `integer` specifying the number of steps to perform in the sampler

    Returns:
      A sampling function that takes random states, and a replcated training state and returns samples as well as
      the number of function evaluations during sampling.
    """
    predictors = [
        get_predictor(predictor if predictor is not None else "NonePredictor")(sde)
        for sde in product.sdes
    ]
    correctors = [
        get_corrector(corrector if corrector is not None else "NoneCorrector")(
            sde, snr, n_steps
        )
        for sde in product.sdes
    ]

    def pc_sampler(rng, xs, N=None, t0=None, tf=None):
        """The PC sampler funciton.

        Args:
          rng: A JAX random state
          state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
        Returns:
          Samples, number of function evaluations
        """

        t0 = product.t0 if t0 is None else t0
        tf = product.tf if tf is None else tf
        t0 = jnp.broadcast_to(t0, xs[0].shape[0])
        tf = jnp.broadcast_to(tf, xs[0].shape[0])

        N = jnp.digitize(tf, bins=jnp.linspace(product.t0, product.tf, product.N)) if N is None else N
        N = jnp.broadcast_to(N, (xs[0].shape[0], 1)) if len(jnp.shape(N)) <= 1 else N

        # Only integrate to eps off the forward start time for numerical stability
        if isinstance(product, ProductRSDE):
            tf = tf + eps
        else:
            t0 = t0 + eps

        # in this discretized code tf and t0 are both scalars so
        # timesteps is 1d and timesteps[i] is a scalar and dt is also a scalar
        # this is in contrast to the continuous PC sampler above
        # so as not to mess up the expected shapes the shapes are maintained
        # here as above, but the resulting arrays are expected to be constant
        timesteps = jnp.linspace(start=t0, stop=tf, num=product.N, endpoint=True)
        dt = (tf - t0) / product.N
        
        def loop_body(i, val):
            rng, x_currs, xs, x_means = val
            t = timesteps[i]
            mask = (N == i)[:, :, None]

            for i, corrector in enumerate(correctors):
                rng, step_rng = random.split(rng)
                x_currs[i], _ = corrector.update_fn(step_rng, x_currs[i], t, dt)

            score = (
                product.score_fn(x_currs, t)
                if isinstance(product, ProductRSDE)
                else None
            )
            for i, predictor in enumerate(predictors):
                rng, step_rng = random.split(rng)
                x_currs[i], x_mean_curr = predictor.update_fn(
                    step_rng,
                    x_currs[i],
                    t,
                    dt,
                    score=score[i] if score is not None else None,
                )

                xs[i] = xs[i] + mask * x_currs[i][:, None, :]
                x_means[i] = x_means[i] + mask * x_mean_curr[:, None, :]

            return rng, x_currs, xs, x_means

        xs_N = [jnp.zeros((x.shape[0], N.shape[1], *x.shape[1:])) for x in xs]
        x_means_N = [jnp.zeros((x.shape[0], N.shape[1], *x.shape[1:])) for x in xs]
        _, _, x, x_mean = jax.lax.fori_loop(
            0, jnp.max(N) + 1, loop_body, (rng, xs, xs_N, x_means_N)
        )

        return inverse_scaler(x_mean if denoise else x), timesteps[
            N.flatten(), jnp.repeat(jnp.arange(0, N.shape[0]), N.shape[1])
        ].reshape(-1, N.shape[1])

    return pc_sampler
