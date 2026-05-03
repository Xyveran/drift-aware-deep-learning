from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class Config:
    """
    Thin wrapper around a nested config dict.

    Supports attribute-style access (cfg.training.lr) and plain dict-style access
    (cfg["training"]["lr"]) interchangeably.

    Usage:
        Load base config only:
            cfg = Config.from_yaml("configs/base.yaml")
        Load base config with a model-specific override:
            cfg = Config.from_yaml("configs/base.yaml", "configs/cnn.yaml")
        Override a single key at runtime:
            cfg.set("training.learning_rate", 5e-4)
    """

    def __init__(self, data: dict[str, Any]) -> None:
        # Wrap nested dicts so attribute access works at all levels.
        self.data = {
            k: Config(v) if isinstance(v, dict) else v for k, v in data.items()
        }

    #
    # Construction
    #

    @classmethod
    def from_yaml(cls, base_path: str | Path, *override_paths: str | Path) -> Config:
        """
        Load a base YAML and apply zero or more override YAMLs on top.

        Keys present in an override file replace those in the base;
        all other base keys are kept. Override files only need to specify
        the keys they change.
        """
        data = cls._load_yaml(base_path)
        for path in override_paths:
            overrides = cls._load_yaml(path)
            data = cls._deep_merge(data, overrides)

        return cls(data)

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively override nested dictionaries into base (override wins on conflicts)."""
        merged = base.copy()

        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Config._deep_merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    #
    # Access
    #

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, dotted_key: str, value: Any) -> None:
        """
        Override a single value using dot notation.

        Ex:
            cfg.set("training.learning_rate", 1e-4)
            cfg.set("drift.detector", "mmd")
        """
        keys = dotted_key.split(".")
        node = self

        for k in keys[:-1]:
            node = node[k]
            if not isinstance(node, Config):
                raise KeyError(f"Intermediate key '{k}' is not a Config section.")

        node._data[keys[-1]] = value

    #
    # Serialization
    #

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain Python dict."""
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in self._data.items()
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write the resolved config to a YAML file (for experiment logging)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
