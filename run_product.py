import os
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm


import numpy as np
import jax
from jax import numpy as jnp
import optax
import haiku as hk

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, call

from score_sde.models.flow import 
from score_sde.losses import get_ema_loss_step_fn
from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection
from score_sde.datasets import random_split, DataLoader, TensorDataset
from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.utils.vis import plot, plot_ref

import seaborn as sns
import matplotlib.pyplot as mplt
import pandas as pd

log = logging.getLogger(__name__)


def run(cfg):
    def train(train_state):
        log.info(f"boundary_enforce: {flow.boundary_enforce}")
        loss = instantiate(
            cfg.loss, pushforward=pushforward, model=model, train=True
        )
        train_step_fn = get_ema_loss_step_fn(loss, optimizer=optimiser, train=True)
        train_step_fn = jax.jit(train_step_fn)

        rng = train_state.rng
        t = tqdm(
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        train_time = timer()
        total_train_time = 0
        mean_loss = 0
        rolling_average = 0  # just the loss
        for step in t:
            data, context = next(train_ds)
            batch = {"data": data, "context": context}
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
            if jnp.isnan(loss).any():
                log.warning("Loss is nan, resetting to checkpoint")
                train_state = restore(ckpt_path)
                # return train_state, False
            else:
                mean_loss = rolling_average * mean_loss + (1 - rolling_average) * loss

            logger.log_metrics({"train/loss": mean_loss}, step)
            t.set_description(f"Loss: {mean_loss:.3f}")

            if step % (2 * cfg.save_freq) == 0 and step != 0:
                save(ckpt_path, train_state, postfix=step)

            if step % (2 * cfg.val_freq) == 0 and step != 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, step
                )
                total_train_time += timer() - train_time

                if cfg.train_val:
                    eval_time = timer()
                    evaluate(train_state, "val", step)
                    logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)
                if cfg.train_plot:
                    plot_time = timer()
                    generate_plots(train_state, "val", step=step)
                    logger.log_metrics(
                        {"plot/time_per_it": (timer() - plot_time)}, step
                    )

                train_time = timer()

        logger.log_metrics({"train/total_time": total_train_time}, step)
        return train_state, True

    def evaluate(train_state, stage, step=None):
        log.info("Running evaluation")
        dataset = eval_ds if stage == "val" else test_ds

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
        likelihood_fn = jax.jit(likelihood_fn)

        logp, nfe, N = 0.0, 0.0, 0
        tot = 0

        if hasattr(dataset, "__len__"):
            for i in range(len(dataset)):
                data, context = next(train_ds)
                logp_step, nfe_step = likelihood_fn(data, context)
                logp += logp_step.sum()
                nfe += nfe_step
                N += logp_step.shape[0]
        else:
            dataset.batch_dims = [cfg.eval_batch_size]
            samples = round(20_000 / cfg.eval_batch_size)
            for i in range(samples):
                batch = next(dataset)
                logp_step, nfe_step = likelihood_fn(*batch)
                logp += logp_step.sum()
                nfe += nfe_step
                N += logp_step.shape[0]
                tot += logp_step.shape[0]
            dataset.batch_dims = [cfg.batch_size]

        logp /= N
        nfe /= len(dataset) if hasattr(dataset, "__len__") else samples

        logger.log_metrics({f"{stage}/logp": logp}, step)
        log.info(f"{stage}/logp = {logp:.3f}")
        logger.log_metrics({f"{stage}/nfe": nfe}, step)
        log.info(f"{stage}/nfe = {nfe:.1f}")

        if stage == "test":  # Estimate normalisation constant
            default_context = context[0] if context is not None else None
            Z = compute_normalization(
                likelihood_fn, data_manifold, context=default_context
            )
            log.info(f"Z = {Z:.2f}")
            logger.log_metrics({f"{stage}/Z": Z}, step)

    def generate_samples(train_state, stage, step=None, M=1000, z=None, context=None):
        log.info("Generating samples (backward process)")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "eval" else test_ds

        # p_0 (backward)
        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        sampler_kwargs = dict(eps=cfg.eps, predictor="GRW")
        sampler = pushforward.get_sampler(
            model_w_dicts, train=False, reverse=True, **sampler_kwargs
        )

        x_true, context = dataset.get_all()

        x_true_dup = z
        rng, next_rng = jax.random.split(rng)
        N = (
            np.vstack(
                [
                    np.ones(z[0].shape[0]) * 0.0 * flow.N,
                    np.ones(z[0].shape[0]) * 0.1 * flow.N,
                    np.ones(z[0].shape[0]) * 0.5 * flow.N,
                    np.ones(z[0].shape[0]) * 0.9 * flow.N,
                    np.ones(z[0].shape[0]) * 1.0 * flow.N,
                ]
            )
            .astype(int)
            .T
        )
        x, t = sampler(next_rng, None, None, N=N, zs=x_true_dup)
        
        jnp.savez(
            f"{ckpt_path}/samples",
            context=context,
            x_true=jnp.concatenate(x_true, axis=-1),
            x=jnp.concatenate(x, axis=-1),
            N=N,
            t=t,
        )
        return context, x_true, x, t

    def generate_noise(train_state, stage, step=None, M=1000):
        log.info("Generating noise (forward process)")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "eval" else test_ds

        x0, context = dataset.get_all()

        prior_fn = jax.jit(
            lambda rng: [data_manifold[i].random_uniform(rng, x0[i].shape[0]) for i in range(len(data_manifold))]
        )
        prior = prior_fn(rng)      


        sampler_kwargs = dict(eps=cfg.eps, predictor="GRW")
        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        
        sampler = pushforward.get_sampler(
            model_w_dicts, train=False, reverse=False, **sampler_kwargs
        )
        z = transform.inv(x0)
        N = (
            np.vstack(
                [
                    np.ones(z[0].shape[0]) * 1,
                    np.ones(z[0].shape[0]) * 0.1 * flow.N,
                    np.ones(z[0].shape[0]) * 0.5 * flow.N,
                    np.ones(z[0].shape[0]) * 0.9 * flow.N,
                    np.ones(z[0].shape[0]) * 1.0 * flow.N,
                ]
            )
            .astype(int)
            .T
        )

        noised, t = sampler(rng, None, context, zs=z, N=N)
        jnp.savez(f"{ckpt_path}/noise",
                  x=jnp.concatenate(x0, axis=-1), 
                  prior=jnp.concatenate(prior, axis=-1), 
                  noised=jnp.concatenate(noised, axis=-1),
                  N=N, t=t)

        return context, prior, z, noised, t

    def generate_plots(train_state, stage, step=None, M=10, sample_prior=False):
        log.info("Generating plots")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "eval" else test_ds

        M = max(1, 10_000 // len(dataset))

        hist_kws = {"stat": "density", "common_norm": False}
        scatter_kws = {"alpha": 0.4}
        

        # p_T (forward)
        # if isinstance(pushforward, SDEPushForward) and sample_prior:
        
        context, prior, z, noised, t = generate_noise(
            train_state, stage, step=step, M=M
        )
        if sample_prior:
            for i in range(t.shape[1]):
                angles = noised[1][:, i, :].reshape(-1, 4, 2)
                angles = np.arctan2(angles[..., 1], angles[..., 0])
                df_noised = pd.DataFrame(
                    data=jnp.concatenate([noised[0][:, i, :], angles], axis=-1)
                )
                angles = prior[1].reshape(-1, 4, 2)
                angles = np.arctan2(angles[..., 1], angles[..., 0])
                df_prior = pd.DataFrame(
                    data=jnp.concatenate([prior[0], angles], axis=-1)
                )
                df_noised['type'] = "Noised (Forward)"
                df_prior['type'] = "Prior"
                df = pd.concat(
                    [df_noised, df_prior], axis=0
                ).reset_index(drop=True)
                plt = sns.pairplot(
                    df, kind='hist', hue="type",
                    plot_kws=scatter_kws, diag_kws=hist_kws
                ).fig

                logger.log_plot(f"Forward Sampling, t={t[0, i]}", plt, step)

            noised = [n[:, -1, :] for n in noised]

        context, x_true, x, t = generate_samples(
            train_state, stage, step=step, M=M, z=prior, context=context
        )
        # prop_in_M = data_manifold.belongs(x[:, -1, :], atol=1e-4).mean()
        # log.info(f"Prop rev samples in M: {100 * prop_in_M.item()}")
        for i in range(t.shape[1]):
            angles = x[1][:, i, :].reshape(-1, 4, 2)
            angles = np.arctan2(angles[..., 1], angles[..., 0])
            df_sample = pd.DataFrame(
                data=jnp.concatenate([x[0][:, i, :], angles], axis=-1)
            )
            plt = sns.pairplot(
                df_sample, kind="hist", plot_kws=scatter_kws, diag_kws=hist_kws
            ).fig
            logger.log_plot(f"Backward Sampling, t={t[0, i]}", plt, step)

        if sample_prior:
            angles = x_true[1].reshape(-1, 4, 2)
            angles = np.arctan2(angles[..., 1], angles[..., 0])
            
            df_true = pd.DataFrame(data=np.hstack([x_true[0], angles]))
            plt = sns.pairplot(
                df_true, kind="hist", plot_kws=scatter_kws, diag_kws=hist_kws
            ).fig
            logger.log_plot(f"Target Distribution", plt, step)

        if noised[0].shape[1] == 2:
            grid = np.linspace(noised.min(axis=0), noised.max(axis=0), 25)
            grid = np.meshgrid(*[grid[:, i] for i in range(noised.shape[1])])
            grid = np.stack(grid, axis=-1).reshape(-1, noised.shape[-1])
            in_manifold = np.all(
                data_manifold.T @ grid.T <= data_manifold.b[:, None], axis=0
            )
            grid = grid[in_manifold]

            model_w_dicts = (model, train_state.params_ema, train_state.model_state)

            score_fn = flow.reparametrise_score_fn(*model_w_dicts)

            for i in range(t.shape[1]):
                grid_score = score_fn(
                    grid, t[0, i] * np.ones(grid.shape[0]), context=None
                )

                for eps in [0, 1e-2]:
                    in_manifold = np.all(
                        data_manifold.T @ grid.T <= data_manifold.b[:, None] - eps,
                        axis=0,
                    )
                    plot_grid, plot_grid_score = (
                        grid[in_manifold],
                        grid_score[in_manifold],
                    )
                    fig, ax = mplt.subplots()
                    q = ax.quiver(
                        plot_grid[:, 0],
                        plot_grid[:, 1],
                        plot_grid_score[:, 0],
                        plot_grid_score[:, 1],
                    )
                    logger.log_plot(f"Score Field, t={t[0, i]}, eps={eps}", fig, step)

    # Main
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    log.info("Stage : Instantiate model")
    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    sdes = [
        instantiate(sde_cfg, manifold=sde_manifold, beta_schedule=sde_beta_schedule)
        for sde_cfg, sde_manifold, sde_beta_schedule in zip(cfg.flow.sdes, data_manifold, beta_schedule)
    ]
    flow = instantiate(cfg.flow, sdes=sdes)

    base = instantiate(cfg.base, model_manifold[0], flow=flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    log.info("Stage : Instantiate dataset")
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng)

    if isinstance(dataset, TensorDataset):
        # split and wrapp dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    log.info("Stage : Instantiate vector field model")

    def model(y, t, context=None):
        """Vector field s_\theta: y, t, context -> T_y M"""
        output_shape = get_class(cfg.generator._target_).output_shape(data_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifolds=data_manifold,
        )
        # TODO: parse context into embedding map
        if context is not None:
            context = (t, context)
        else:
            context = t
        return score(y, context)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    t = jnp.zeros((cfg.batch_size, 1))
    data, context = next(train_ds)
    params, state = model.init(
        rng=next_rng, y=transform.inv(data), t=t, context=context
    )

    log.info("Stage : Instantiate optimiser")
    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(
        instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn)
    )
    opt_state = optimiser.init(params)

    if cfg.resume or cfg.mode == "test":  # if resume or evaluate
        log.info("Loading from saved model")
        train_state = restore(ckpt_path)
        print(train_state)
    else:
        rng, next_rng = jax.random.split(rng)
        train_state = TrainState(
            opt_state=opt_state,
            model_state=state,
            step=0,
            params=params,
            ema_rate=cfg.ema_rate,
            params_ema=params,
            rng=next_rng,  # TODO: we should actually use this for reproducibility
        )
        save(ckpt_path, train_state)

    if cfg.mode == "train" or cfg.mode == "all":
        if train_state.step == 0 and cfg.test_test:
            evaluate(train_state, "test", step=cfg.steps)
        if train_state.step == 0 and cfg.test_plot:
            generate_plots(train_state, "test", step=0, sample_prior=True)
        log.info("Stage : Training")
        train_state, success = train(train_state)
    if cfg.mode == "test" or (cfg.mode == "all" and success):
        log.info("Stage : Test")
        if cfg.test_val:
            evaluate(train_state, "val")
        if cfg.test_test:
            evaluate(train_state, "test")
        if cfg.test_plot:
            generate_plots(train_state, "test")
        if cfg.test_plot:
            generate_samples(train_state, "test")
            generate_noise(train_state, "test")

        success = True
    logger.save()
    logger.finalize("success" if success else "failure")
