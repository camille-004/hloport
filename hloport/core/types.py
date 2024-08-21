from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray


class ActivationType(Enum):
    RELU = auto()
    TANH = auto()
    SIGMOID = auto()


class InitType(Enum):
    HE_NORMAL = auto()
    GLOROT_UNIFORM = auto()
    ZEROS = auto()
    ONES = auto()


@runtime_checkable
class Activation(Protocol):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: ...


class Layer(NamedTuple):
    weights: jnp.ndarray
    biases: jnp.ndarray


InitFn = Callable[[PRNGKeyArray, tuple[int, ...]], jnp.ndarray]


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    activation: ActivationType = ActivationType.RELU
    weight_init: InitType = InitType.HE_NORMAL
    bias_init: InitType = InitType.ZEROS
