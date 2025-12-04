# LongLive Benchmark Config Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark configuration."""

import pytest
import torch
import tempfile
import os
import json
import yaml

from benchmarks.config import (
    BenchmarkConfig,
    BENCHMARK_CONFIGS,
    create_optimization_sweep,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BenchmarkConfig()

        assert config.model_name == "Wan2.1-T2V-1.3B"
        assert config.local_attn_size == 12
        assert config.sink_size == 3
        assert config.num_output_frames == 120
        assert config.batch_size == 1
        assert config.warmup_iterations == 2
        assert config.benchmark_iterations == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            num_output_frames=60,
            batch_size=2,
            use_fp8=True,
        )

        assert config.num_output_frames == 60
        assert config.batch_size == 2
        assert config.use_fp8 is True

    def test_denoising_steps_default(self):
        """Test default denoising steps."""
        config = BenchmarkConfig()
        assert config.denoising_steps == [1000, 750, 500, 250]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BenchmarkConfig(
            num_output_frames=60,
            use_torch_compile=True,
        )
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["num_output_frames"] == 60
        assert d["use_torch_compile"] is True
        assert "model_name" in d

    def test_to_yaml(self):
        """Test YAML export."""
        config = BenchmarkConfig(num_output_frames=90)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            path = f.name

        try:
            config.to_yaml(path)
            assert os.path.exists(path)

            with open(path) as f:
                data = yaml.safe_load(f)

            assert data["num_output_frames"] == 90
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_from_yaml(self):
        """Test YAML import."""
        config_data = {
            "model_name": "TestModel",
            "num_output_frames": 48,
            "batch_size": 2,
            "warmup_iterations": 1,
            "benchmark_iterations": 5,
            "denoising_steps": [1000, 500],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = BenchmarkConfig.from_yaml(path)
            assert config.model_name == "TestModel"
            assert config.num_output_frames == 48
            assert config.batch_size == 2
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_from_json(self):
        """Test JSON import."""
        config_data = {
            "model_name": "JSONModel",
            "num_output_frames": 36,
            "batch_size": 1,
            "warmup_iterations": 2,
            "benchmark_iterations": 3,
            "denoising_steps": [1000, 750, 500, 250],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            path = f.name

        try:
            config = BenchmarkConfig.from_json(path)
            assert config.model_name == "JSONModel"
            assert config.num_output_frames == 36
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_populate_hardware_info(self):
        """Test hardware info population."""
        config = BenchmarkConfig()
        config.populate_hardware_info()

        assert config.torch_version != ""
        assert config.torch_version == torch.__version__

        if torch.cuda.is_available():
            assert config.gpu_name != ""
            assert config.gpu_name != "N/A (CPU)"
        else:
            assert config.gpu_name == "N/A (CPU)"

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = BenchmarkConfig(
            num_output_frames=120,
            num_frame_per_block=3,
            batch_size=1,
            warmup_iterations=2,
            benchmark_iterations=5,
        )
        # Should not raise
        config.validate()

    def test_validate_invalid_frames(self):
        """Test validation fails for invalid frame count."""
        config = BenchmarkConfig(
            num_output_frames=121,  # Not divisible by 3
            num_frame_per_block=3,
        )
        with pytest.raises(AssertionError):
            config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch size."""
        config = BenchmarkConfig(batch_size=0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_validate_invalid_iterations(self):
        """Test validation fails for invalid iteration counts."""
        config = BenchmarkConfig(benchmark_iterations=0)
        with pytest.raises(AssertionError):
            config.validate()

        config = BenchmarkConfig(warmup_iterations=-1)
        with pytest.raises(AssertionError):
            config.validate()

    def test_validate_local_attn_size(self):
        """Test validation of local attention size."""
        # Valid: -1 (global attention)
        config = BenchmarkConfig(local_attn_size=-1)
        config.validate()

        # Valid: positive
        config = BenchmarkConfig(local_attn_size=12)
        config.validate()

        # Invalid: 0
        config = BenchmarkConfig(local_attn_size=0)
        with pytest.raises(AssertionError):
            config.validate()


class TestPredefinedConfigs:
    """Tests for predefined benchmark configurations."""

    def test_standard_config(self):
        """Test standard configuration."""
        config = BENCHMARK_CONFIGS["standard"]
        assert config.num_output_frames == 120
        assert config.batch_size == 1
        assert config.benchmark_iterations == 10

    def test_quick_config(self):
        """Test quick configuration."""
        config = BENCHMARK_CONFIGS["quick"]
        assert config.num_output_frames == 24
        assert config.warmup_iterations == 1
        assert config.benchmark_iterations == 3

    def test_long_video_config(self):
        """Test long video configuration."""
        config = BENCHMARK_CONFIGS["long_video"]
        assert config.num_output_frames == 480

    def test_fp8_config(self):
        """Test FP8 configuration."""
        config = BENCHMARK_CONFIGS["fp8_test"]
        assert config.use_fp8 is True

    def test_compiled_config(self):
        """Test compiled configuration."""
        config = BENCHMARK_CONFIGS["compiled"]
        assert config.use_torch_compile is True
        assert config.compile_mode == "reduce-overhead"

    def test_all_configs_valid(self):
        """Test all predefined configs are valid."""
        for name, config in BENCHMARK_CONFIGS.items():
            try:
                config.validate()
            except AssertionError as e:
                pytest.fail(f"Config '{name}' failed validation: {e}")


class TestOptimizationSweep:
    """Tests for optimization sweep creation."""

    def test_creates_multiple_configs(self):
        """Test sweep creates multiple configurations."""
        configs = create_optimization_sweep()
        assert len(configs) > 0

    def test_sweep_has_baseline(self):
        """Test sweep includes baseline config."""
        configs = create_optimization_sweep()
        names = [name for name, _ in configs]
        assert "baseline" in names

    def test_sweep_has_torch_compile(self):
        """Test sweep includes torch.compile config."""
        configs = create_optimization_sweep()
        names = [name for name, _ in configs]
        assert "torch_compile" in names

    def test_sweep_has_window_variations(self):
        """Test sweep includes window size variations."""
        configs = create_optimization_sweep()
        names = [name for name, _ in configs]

        assert "window_6" in names
        assert "window_9" in names
        assert "window_12" in names

    def test_sweep_has_fp8(self):
        """Test sweep includes FP8 config."""
        configs = create_optimization_sweep()
        names = [name for name, _ in configs]
        assert "fp8" in names

    def test_sweep_configs_have_hardware_info(self):
        """Test sweep configs have hardware info populated."""
        configs = create_optimization_sweep()

        for name, config in configs:
            assert config.torch_version != ""

    def test_sweep_torch_compile_config(self):
        """Test torch.compile config in sweep."""
        configs = create_optimization_sweep()
        config_dict = {name: cfg for name, cfg in configs}

        assert config_dict["torch_compile"].use_torch_compile is True
        assert config_dict["torch_compile"].compile_mode == "reduce-overhead"

    def test_sweep_fp8_config(self):
        """Test FP8 config in sweep."""
        configs = create_optimization_sweep()
        config_dict = {name: cfg for name, cfg in configs}

        assert config_dict["fp8"].use_fp8 is True
