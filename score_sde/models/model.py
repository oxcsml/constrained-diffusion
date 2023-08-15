"""Modified code from https://github.com/yang-song/score_sde"""
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
# from collections import Iterable

import jax
import jax.numpy as jnp
from score_sde.utils.jax import batch_mul
from score_sde.utils.typing import ParametrisedScoreFunction


def get_score_fn(
    sde,
    model: ParametrisedScoreFunction,
    params,
    state,
    train=False,
    return_state=False,
    std_trick=False,
    residual_trick=True,
    boundary_enforce=False,
    boundary_dis=0.01,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
      sde: An `sde.SDE` object that represents the forward SDE.
      model: A Haiku transformed function representing the score function model
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      return_state: If `True`, return the new mutable states alongside the model output.
    Returns:
      A score function.
    """

    def smoother_fn(bounded_manifold, y, eps=1e-6, maximum=1):
        return jnp.minimum(
            jax.nn.relu(bounded_manifold.distance_to_boundary(y) - boundary_dis) + eps,
            maximum,
        )

    def score_fn(y, t, context, rng=None):
        model_out, new_state = model.apply(
            params, state, rng, y=y, t=t, context=context
        )
        score = model_out

        if std_trick:
            # NOTE: scaling the output with 1.0 / std helps cf 'Improved Techniques for Training Score-Based Generative Model'
            # stds = [sde.sdes[i].marginal_prob(jnp.zeros_like(score[i]), t)[1] for i in range(len(sde.sdes))]
            # score = [batch_mul(score[i], 1.0 / stds[i]) for i in range(len(sde.sdes))]
            std = sde.marginal_prob(jnp.zeros_like(y), t)[1]
            score = batch_mul(score, 1.0 / std)
        if residual_trick:
            # NOTE: so that if NN = 0 then time reversal = forward
            fwd_drift = sde.drift(y, t)
            residual = 2 * fwd_drift / sde.beta_schedule.beta_t(t)[..., None]
            score += residual
        if boundary_enforce:
            if isinstance(score, tuple):
                bounded_manifold = (
                    sde.sdes[0].manifold if hasattr(sde, "sdes") else sde.manifold
                )
                if isinstance(y, list):
                    smooth_value = smoother_fn(bounded_manifold, y[0])
                    if len(smooth_value.shape) == 1:
                        smooth_value = smooth_value[..., None]
                    score = (
                        score[0] * smooth_value,
                        score[1],
                    )
                else:
                    smooth_value = smoother_fn(
                        bounded_manifold, y[..., : bounded_manifold.dim]
                    )
                    if len(smooth_value.shape) == 1:
                        smooth_value = smooth_value[..., None]
                    score = (
                        score[0] * smooth_value,
                        score[1],
                    )
            else:
                smooth_value = smoother_fn(sde.manifold, y)
                if len(smooth_value.shape) == 1:
                    smooth_value = smooth_value[..., None]
                score *= smooth_value

        if return_state:
            return score, new_state
        else:
            return score

    return score_fn
