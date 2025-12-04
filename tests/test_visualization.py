# LongLive Visualization Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark visualization."""

import pytest
import json
import tempfile
import os

from benchmarks.visualization import (
    generate_time_budget_report,
    generate_optimization_comparison_chart,
)


class TestGenerateTimeBudgetReport:
    """Tests for time budget report generation."""

    def test_text_format(self, mock_benchmark_results):
        """Test text format report generation."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="text"
        )

        assert "LONGLIVE TIME BUDGET BREAKDOWN" in report
        assert "Configuration:" in report
        assert "PERFORMANCE SUMMARY" in report

    def test_text_contains_metrics(self, mock_benchmark_results):
        """Test text report contains metrics."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="text"
        )

        assert "FPS:" in report
        assert "Latency per frame:" in report
        assert "Steady-state:" in report

    def test_text_contains_breakdown(self, mock_benchmark_results):
        """Test text report contains time breakdown."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="text"
        )

        assert "Initialization" in report
        assert "Diffusion" in report
        assert "VAE Decode" in report

    def test_text_target_assessment(self, mock_benchmark_results):
        """Test text report contains target assessment."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="text"
        )

        assert "TARGET ASSESSMENT" in report
        # With 50ms latency, should fail 40ms target
        assert "[FAIL]" in report

    def test_text_target_pass(self):
        """Test text report shows pass for good latency."""
        results = {
            "config": {"model_name": "Test"},
            "summary": {
                "fps": {"mean": 30.0, "std": 1.0},
                "ms_per_frame": {"mean": 33.0, "std": 2.0, "p95": 35.0},
                "memory_peak_gb": {"mean": 8.0},
                "steady_state_inter_frame_ms": {"mean": 32.0},
            }
        }
        report = generate_time_budget_report(results, format="text")
        assert "[PASS]" in report

    def test_json_format(self, mock_benchmark_results):
        """Test JSON format report generation."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="json"
        )

        data = json.loads(report)

        assert "summary" in data
        assert "config" in data
        assert "time_budget" in data
        assert "target_assessment" in data

    def test_json_time_budget(self, mock_benchmark_results):
        """Test JSON report has time budget breakdown."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="json"
        )

        data = json.loads(report)
        time_budget = data["time_budget"]

        assert "initialization_pct" in time_budget
        assert "diffusion_pct" in time_budget
        assert "vae_decode_pct" in time_budget

    def test_json_target_assessment(self, mock_benchmark_results):
        """Test JSON report has target assessment."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="json"
        )

        data = json.loads(report)
        assert data["target_assessment"]["target_ms"] == 40.0
        assert "meets_target" in data["target_assessment"]

    def test_html_format(self, mock_benchmark_results):
        """Test HTML format report generation."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="html"
        )

        assert "<!DOCTYPE html>" in report
        assert "<title>LongLive Time Budget</title>" in report
        assert "plotly" in report.lower()

    def test_html_contains_metrics(self, mock_benchmark_results):
        """Test HTML report contains key metrics."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="html"
        )

        assert "FPS" in report
        assert "ms/frame" in report
        assert "Peak GB" in report

    def test_html_contains_charts(self, mock_benchmark_results):
        """Test HTML report contains chart divs."""
        report = generate_time_budget_report(
            mock_benchmark_results,
            format="html"
        )

        assert 'id="pie-chart"' in report
        assert 'id="box-chart"' in report

    def test_invalid_format(self, mock_benchmark_results):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Unknown format"):
            generate_time_budget_report(
                mock_benchmark_results,
                format="invalid"
            )

    def test_output_to_file(self, mock_benchmark_results):
        """Test writing report to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            path = f.name

        try:
            report = generate_time_budget_report(
                mock_benchmark_results,
                output_path=path,
                format="text"
            )

            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "LONGLIVE TIME BUDGET BREAKDOWN" in content
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_empty_summary(self):
        """Test handling of empty summary."""
        results = {"config": {}, "summary": {}}
        report = generate_time_budget_report(results, format="text")
        # Should not raise, but have default values
        assert "LONGLIVE TIME BUDGET BREAKDOWN" in report


class TestGenerateOptimizationComparisonChart:
    """Tests for optimization comparison chart generation."""

    def test_creates_html_file(self):
        """Test that HTML file is created."""
        results_list = [
            {"config_name": "baseline", "fps_mean": 20.0, "ms_per_frame_mean": 50.0, "memory_peak_gb": 8.0},
            {"config_name": "fp8", "fps_mean": 25.0, "ms_per_frame_mean": 40.0, "memory_peak_gb": 6.0},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            path = f.name

        try:
            generate_optimization_comparison_chart(results_list, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_html_contains_configs(self):
        """Test HTML contains configuration names."""
        results_list = [
            {"config_name": "baseline", "fps_mean": 20.0, "ms_per_frame_mean": 50.0, "memory_peak_gb": 8.0},
            {"config_name": "optimized", "fps_mean": 30.0, "ms_per_frame_mean": 33.0, "memory_peak_gb": 7.0},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            path = f.name

        try:
            generate_optimization_comparison_chart(results_list, path)

            with open(path) as f:
                content = f.read()

            assert "baseline" in content
            assert "optimized" in content
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_html_contains_charts(self):
        """Test HTML contains chart elements."""
        results_list = [
            {"config_name": "test", "fps_mean": 20.0, "ms_per_frame_mean": 50.0, "memory_peak_gb": 8.0},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            path = f.name

        try:
            generate_optimization_comparison_chart(results_list, path)

            with open(path) as f:
                content = f.read()

            assert 'id="fps-chart"' in content
            assert 'id="latency-chart"' in content
            assert "plotly" in content.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_handles_missing_config_name(self):
        """Test handling of results without config_name."""
        results_list = [
            {"fps_mean": 20.0, "ms_per_frame_mean": 50.0, "memory_peak_gb": 8.0},
            {"fps_mean": 25.0, "ms_per_frame_mean": 40.0, "memory_peak_gb": 6.0},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            path = f.name

        try:
            # Should not raise
            generate_optimization_comparison_chart(results_list, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
