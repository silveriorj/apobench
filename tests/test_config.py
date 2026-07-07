"""Tests for configuration system."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from pof.config.loader import load_config, _deep_merge
from pof.config.schemas import RunConfig, LLMConfig, EvalConfig
from pof.core.exceptions import ConfigError


class TestSchemas:
    def test_default_config(self):
        config = RunConfig()
        assert config.llm.backend == "huggingface"
        assert config.evaluation.sample_size == 50
        assert config.optimizer.method == "see"
        assert config.seed == 42

    def test_llm_config(self):
        config = LLMConfig(model_name="gpt-4", backend="openai")
        assert config.model_name == "gpt-4"
        assert config.backend == "openai"

    def test_eval_config(self):
        config = EvalConfig(sample_size=100, racing_enabled=False)
        assert config.sample_size == 100
        assert config.racing_enabled is False


class TestLoader:
    def test_load_defaults(self):
        config = load_config()
        assert isinstance(config, RunConfig)
        assert config.llm.backend == "huggingface"

    def test_load_json(self, tmp_path):
        config_data = {
            "llm": {"model_name": "test-model", "backend": "openai"},
            "optimizer": {"method": "swift"},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = load_config(path)
        assert config.llm.model_name == "test-model"
        assert config.optimizer.method == "swift"

    def test_load_yaml(self, tmp_path):
        yaml_content = """
llm:
  model_name: yaml-model
  backend: huggingface
optimizer:
  method: apex
  population_size: 10
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        config = load_config(path)
        assert config.llm.model_name == "yaml-model"
        assert config.optimizer.method == "apex"
        assert config.optimizer.population_size == 10

    def test_overrides(self, tmp_path):
        config_data = {"llm": {"model_name": "base-model"}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = load_config(path, overrides={"llm": {"model_name": "override-model"}})
        assert config.llm.model_name == "override-model"

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_MODEL", "env-model")
        config_data = {"llm": {"model_name": "${TEST_MODEL}"}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = load_config(path)
        assert config.llm.model_name == "env-model"

    def test_missing_file(self):
        with pytest.raises(ConfigError):
            load_config("nonexistent.yaml")


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"llm": {"model": "a", "device": "cpu"}}
        override = {"llm": {"model": "b"}}
        result = _deep_merge(base, override)
        assert result == {"llm": {"model": "b", "device": "cpu"}}