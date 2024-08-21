from functools import wraps
from typing import Any, Callable, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray

from hloport.core.types import (
    Activation,
    ActivationType,
    InitFn,
    InitType,
    Layer,
)

T = TypeVar("T")


def factory(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(type_enum: Any) -> T:
        try:
            return func(type_enum)
        except KeyError:
            raise ValueError(f"Unsupported type: {type_enum}")

    return wrapper


@factory
def get_activation(activation_type: ActivationType) -> Activation:
    activation_map: dict[ActivationType, Activation] = {
        ActivationType.RELU: jax.nn.relu,
        ActivationType.TANH: cast(Activation, jnp.tanh),
        ActivationType.SIGMOID: jax.nn.sigmoid,
    }
    return activation_map[activation_type]


@factory
def get_param_init(init_type: InitType) -> Callable:
    def he_normal(key: PRNGKeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
        return random.normal(key, shape) * jnp.sqrt(2.0 / shape[0])

    def glorot_uniform(
        key: PRNGKeyArray, shape: tuple[int, ...]
    ) -> jnp.ndarray:
        limit = jnp.sqrt(6 / (shape[0] + shape[1]))
        return random.uniform(key, shape, minval=-limit, maxval=limit)

    init_map: dict[InitType, InitFn] = {
        InitType.HE_NORMAL: he_normal,
        InitType.GLOROT_UNIFORM: glorot_uniform,
        InitType.ZEROS: lambda key, shape: jnp.zeros(shape),
        InitType.ONES: lambda key, shape: jnp.ones(shape),
    }
    return init_map[init_type]


def init_layers(
    key: jax.Array,
    dims: list[int],
    weight_init_type: InitType,
    bias_init_type: InitType,
) -> list[Layer]:
    weight_init = get_param_init(weight_init_type)
    bias_init = get_param_init(bias_init_type)

    layers = []
    for i in range(len(dims) - 1):
        key, w_key, b_key = random.split(key, 3)
        weights = weight_init(w_key, (dims[i], dims[i + 1]))
        biases = bias_init(b_key, (dims[i + 1],))
        layers.append(Layer(weights, biases))
    return layers
