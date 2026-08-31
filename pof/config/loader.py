"""Configuration loader — YAML/JSON with environment variable substitution."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

from pof.config.schemas import RunConfig
from pof.core.exceptions import ConfigError


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = pattern.findall(value)
        for var_name in matches:
            env_val = os.environ.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_val)
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> RunConfig:
    """Load configuration from YAML/JSON file with env-var substitution.

    Args:
        path: Path to config file (YAML or JSON). If None, returns defaults.
        overrides: Dictionary of overrides to apply on top of loaded config.

    Returns:
        Validated RunConfig instance.

    Raises:
        ConfigError: If file not found or validation fails.
    """
    data: Dict[str, Any] = {}

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ConfigError(f"Unsupported config format: {path.suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse config file: {e}") from e

    # Substitute environment variables
    data = _substitute_env_vars(data)

    # Apply overrides
    if overrides:
        data = _deep_merge(data, overrides)

    # Validate with Pydantic
    try:
        return RunConfig(**data)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result