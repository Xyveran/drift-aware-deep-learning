from __future__ import annotations

import torch.nn as nn
from src.models.cnn import CNN
from src.utils.config import Config

#
# Registry
#

# Maps model name strings (from base.yaml / model-specific overrides) to their
# from_config constructors. Registering here is the only change needed when a
# new architecture is added.
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn": CNN,
}


#
# Factory
#


def build_model(cfg: Config) -> nn.Module:
    """
    Instantiate and return a model from the project Config.

    Reads ``cfg.model.name`` to select the architecture, then delegates to
    that model's ``from_config`` classmethod for construction.

    Parameters
        cfg : Config
            Fully merged config (base.yaml + model-specific override).
            Must contain a ``model.name`` key matching a registered architecture.

    Returns
        nn.Module
            Instantiated model, ready for training.

    Raises
        ValueError
            If ``cfg.model.name`` is not found in the model registry.

    Example
        >>> cfg = Config.from_yaml("configs/base.yaml", "configs/cnn.yaml")
        >>> model = build_model(cfg)
        >>> print(model)
        CNN(input_channels=128, hidden_channels=[64, 128, 256], num_classes=6, params=...)
    """
    name = cfg.model.name.lower()

    if name not in _MODEL_REGISTRY:
        registered = sorted(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Registered models: {registered}. "
            f"Add it to _MODEL_REGISTRY in factory.py."
        )

    model_cls = _MODEL_REGISTRY[name]
    return model_cls.from_config(cfg)


#
# Registry helpers
#


def register_model(name: str, model_cls: type[nn.Module]) -> None:
    """
    Register a new model class under the given name.

    Intended for use in experiments or ablations that define custom
    architectures without modifying factory.py directly.

    Example
        >>> from src.models.factory import register_model
        >>> from my_experiment import ThinCNN
        >>> register_model("thin_cnn", ThinCNN)
        >>> cfg.set("model.name", "thin_cnn")
        >>> model = build_model(cfg)
    """
    if name in _MODEL_REGISTRY:
        raise ValueError(
            f"A model named '{name}' is already registered. "
            f"Use a different name or remove the existing entry first."
        )
    _MODEL_REGISTRY[name] = model_cls


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return sorted(_MODEL_REGISTRY.keys())
