from jax import random

from hloport.core.types import ActivationType, InitType, ModelConfig
from hloport.models.simple_model import SimpleModel


def main() -> None:
    config = ModelConfig(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=5,
        activation=ActivationType.RELU,
        weight_init=InitType.HE_NORMAL,
        bias_init=InitType.ZEROS,
    )

    model = SimpleModel(config)
    print(f"Model created with {model.param_count} parameters.")

    key = random.PRNGKey(0)
    x = random.normal(key, (1, config.input_dim))

    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    main()
