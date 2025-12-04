# LongLive Benchmark Runner Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark runner (mocked tests)."""

import pytest
import json
import csv
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from benchmarks.config import BenchmarkConfig
from benchmarks.runner import BenchmarkRunner


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        assert runner.config is config
        assert runner.pipeline is None
        assert runner.device is None
        assert len(runner.results) == 0

    def test_init_validates_config(self):
        """Test that config is validated on init."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate') as mock_validate:
                runner = BenchmarkRunner(config)
                mock_validate.assert_called_once()

    def test_init_populates_hardware_info(self):
        """Test that hardware info is populated on init."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info') as mock_populate:
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)
                mock_populate.assert_called_once()


class TestBenchmarkRunnerExport:
    """Tests for result export functionality."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for export tests."""
        return {
            "config": {"model_name": "Test", "num_output_frames": 120},
            "summary": {
                "fps": {"mean": 20.0, "std": 1.0},
                "ms_per_frame": {"mean": 50.0, "std": 2.0, "p50": 49.0, "p95": 54.0, "p99": 55.0},
                "memory_peak_gb": {"mean": 8.0, "max": 8.5},
                "steady_state_inter_frame_ms": {"mean": 48.0},
            },
            "iterations": [
                {"iteration": 0, "fps": 19.5, "ms_per_frame": 51.3, "is_warmup": False},
                {"iteration": 1, "fps": 20.2, "ms_per_frame": 49.5, "is_warmup": False},
                {"iteration": 2, "fps": 20.1, "ms_per_frame": 49.8, "is_warmup": False},
            ],
            "timestamp": "2024-01-01T00:00:00",
        }

    def test_export_json(self, sample_results):
        """Test JSON export."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            runner.export_json(path, sample_results)
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)

            assert data["config"]["model_name"] == "Test"
            assert len(data["iterations"]) == 3
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_export_csv(self, sample_results):
        """Test CSV export."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            runner.export_csv(path, sample_results)
            assert os.path.exists(path)

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert "iteration" in rows[0]
            assert "fps" in rows[0]
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_export_csv_empty_iterations(self, sample_results):
        """Test CSV export with no iterations."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        sample_results["iterations"] = []

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            # Should not raise
            runner.export_csv(path, sample_results)
            # File may not be created if no iterations
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestBenchmarkRunnerPrintSummary:
    """Tests for summary printing."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for print tests."""
        return {
            "config": {
                "model_name": "Test",
                "num_output_frames": 120,
                "batch_size": 1,
                "local_attn_size": 12,
                "denoising_steps": [1000, 750, 500, 250],
                "use_torch_compile": True,
                "use_fp8": False,
            },
            "summary": {
                "fps": {"mean": 20.7, "std": 0.5, "min": 19.8, "max": 21.2},
                "ms_per_frame": {"mean": 48.3, "std": 1.2, "p50": 48.0, "p95": 50.1, "p99": 51.0},
                "memory_peak_gb": {"mean": 8.0, "max": 8.5},
                "steady_state_inter_frame_ms": {"mean": 46.0},
            },
        }

    def test_print_summary_pass(self, sample_results, capsys):
        """Test summary printing when passing target."""
        # Modify to pass target
        sample_results["summary"]["ms_per_frame"]["mean"] = 35.0

        config = BenchmarkConfig()
        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        runner.print_summary(sample_results)
        captured = capsys.readouterr()

        assert "BENCHMARK SUMMARY" in captured.out
        assert "[PASS]" in captured.out

    def test_print_summary_fail(self, sample_results, capsys):
        """Test summary printing when failing target."""
        config = BenchmarkConfig()
        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        runner.print_summary(sample_results)
        captured = capsys.readouterr()

        assert "BENCHMARK SUMMARY" in captured.out
        assert "[FAIL]" in captured.out

    def test_print_summary_contains_metrics(self, sample_results, capsys):
        """Test summary contains all key metrics."""
        config = BenchmarkConfig()
        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        runner.print_summary(sample_results)
        captured = capsys.readouterr()

        assert "FPS:" in captured.out
        assert "ms/frame:" in captured.out
        assert "Steady-state:" in captured.out
        assert "Peak memory:" in captured.out


class TestBenchmarkRunnerComputeSummary:
    """Tests for summary computation."""

    def test_compute_summary_basic(self):
        """Test basic summary computation."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        results = [
            {"total_time_ms": 5000, "fps": 20.0, "ms_per_frame": 50.0, "memory_peak_gb": 8.0, "steady_state_inter_frame_ms": 48.0},
            {"total_time_ms": 4800, "fps": 21.0, "ms_per_frame": 47.6, "memory_peak_gb": 8.2, "steady_state_inter_frame_ms": 46.0},
            {"total_time_ms": 5200, "fps": 19.5, "ms_per_frame": 51.3, "memory_peak_gb": 7.8, "steady_state_inter_frame_ms": 50.0},
        ]

        summary = runner._compute_summary(results)

        assert "total_time_ms" in summary
        assert "fps" in summary
        assert "ms_per_frame" in summary
        assert "memory_peak_gb" in summary
        assert "steady_state_inter_frame_ms" in summary

        assert summary["fps"]["mean"] == pytest.approx(20.17, rel=0.01)
        assert summary["num_iterations"] == 3

    def test_compute_summary_statistics(self):
        """Test that all statistics are computed."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        results = [
            {"total_time_ms": 5000, "fps": 20.0, "ms_per_frame": 50.0, "memory_peak_gb": 8.0, "steady_state_inter_frame_ms": 48.0}
            for _ in range(10)
        ]

        summary = runner._compute_summary(results)

        # Check all required stats are present
        assert "mean" in summary["total_time_ms"]
        assert "std" in summary["total_time_ms"]
        assert "p50" in summary["total_time_ms"]
        assert "p95" in summary["total_time_ms"]
        assert "p99" in summary["total_time_ms"]
        assert "min" in summary["total_time_ms"]
        assert "max" in summary["total_time_ms"]


class TestBenchmarkRunnerGenerateInputs:
    """Tests for input generation."""

    def test_deterministic_inputs(self):
        """Test that inputs are deterministic with same seed."""
        config = BenchmarkConfig(seed=42)

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        runner.device = "cpu"

        noise1, prompts1 = runner._generate_inputs(42)
        noise2, prompts2 = runner._generate_inputs(42)

        assert torch.allclose(noise1, noise2)
        assert prompts1 == prompts2

    def test_different_seeds_different_inputs(self):
        """Test that different seeds produce different inputs."""
        config = BenchmarkConfig()

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        runner.device = "cpu"

        noise1, _ = runner._generate_inputs(42)
        noise2, _ = runner._generate_inputs(43)

        assert not torch.allclose(noise1, noise2)

    def test_input_shape(self):
        """Test generated input shapes."""
        config = BenchmarkConfig(
            batch_size=2,
            num_output_frames=60,
            latent_channels=16,
            latent_height=60,
            latent_width=104,
        )

        with patch.object(config, 'populate_hardware_info'):
            with patch.object(config, 'validate'):
                runner = BenchmarkRunner(config)

        runner.device = "cpu"

        noise, prompts = runner._generate_inputs(0)

        assert noise.shape == (2, 60, 16, 60, 104)
        assert len(prompts) == 2


# Import torch for test fixtures
import torch
