from dataclasses import dataclass
from collections.abc import Iterable

import jax
import haiku as hk
import jax.numpy as jnp

from .layers import get_activation
from score_sde.utils import register_model


@register_model
@dataclass
class MLP:
    hidden_shapes: list
    output_shape: list
    act: str
    bias: bool = True

    def __call__(self, x):
        for hs in self.hidden_shapes:
            x = hk.Linear(output_size=hs, with_bias=self.bias)(x)
            x = get_activation(self.act)(x)

        x = [hk.Linear(output_size=s)(x) for s in self.output_shape] \
            if isinstance(self.output_shape, Iterable) else hk.Linear(output_size=self.output_shape)(x)

        return x


@register_model
@dataclass
class MHA(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act, heads=10):
        super().__init__()
        self._len = MLP(
            hidden_shapes=hidden_shapes[:-1], output_shape=hidden_shapes[-1], act=act
        )
        self._key = MLP(
            hidden_shapes=hidden_shapes[:-1], output_shape=hidden_shapes[-1], act=act
        )
        self._query = MLP(
            hidden_shapes=hidden_shapes[:-1], output_shape=hidden_shapes[-1], act=act
        )
        self._value = MLP(
            hidden_shapes=hidden_shapes[:-1], output_shape=hidden_shapes[-1], act=act
        )
        self._out = MLP(hidden_shapes=hidden_shapes, output_shape=1, act=act)
        self._mha = hk.MultiHeadAttention(3, heads, w_init_scale=1)

    def __call__(self, Q, K, V):

        LQ = self._len(jnp.arange(Q.shape[-2])[None, :])
        print(LQ)
        EQ = self._query(jnp.concatenate([Q, LQ], axis=-2))
        LK = self._len(jnp.arange(K.shape[-2])[None, :])
        EK = self._key(jnp.concatenate([K, LK], axis=-2))
        LV = self._len(jnp.arange(V.shape[-2])[None, :])
        EV = self._value(jnp.concatenate([V, LV], axis=-2))

        x = self._mha(EQ, EK, EV)
        x = self._out(x.reshape(*x.shape[:-1], 3, -1))[..., 0]
        return x
