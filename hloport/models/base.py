from abc import ABC, abstractmethod

import jax.numpy as jnp

from hloport.core.types import ModelConfig


class BaseModel(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @property
    @abstractmethod
    def param_count(self) -> int:
        pass
