from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from hloport.core.types import Layer, ModelConfig
from hloport.models.initialization import get_activation, init_layers

from .base import BaseModel


class SimpleModel(BaseModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.layers = self._init_layers()
        self.activation = get_activation(self.config.activation)
        self._apply_jit = jax.jit(self._apply)

    def _init_layers(self) -> list[Layer]:
        key = random.PRNGKey(0)  # type: ignore
        dims = (
            [self.config.input_dim]
            + self.config.hidden_dims
            + [self.config.output_dim]
        )
        return init_layers(
            key, dims, self.config.weight_init, self.config.bias_init
        )

    def _apply(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.activation(jnp.dot(x, layer.weights) + layer.biases)
        return jnp.dot(x, self.layers[-1].weights) + self.layers[-1].biases

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._apply_jit(x)

    @property
    def param_count(self) -> int:
        return sum(w.size + b.size for w, b in self.layers)
