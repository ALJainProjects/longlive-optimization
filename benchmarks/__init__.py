# LongLive Benchmarks Package
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark infrastructure for LongLive video generation.

This package provides:
- BenchmarkConfig: Reproducible benchmark configuration
- BenchmarkRunner: Benchmark harness with warmup and statistics
- Visualization tools for time budget reports
- Optimization sweep utilities
"""

from benchmarks.config import BenchmarkConfig, BENCHMARK_CONFIGS, create_optimization_sweep
from benchmarks.runner import BenchmarkRunner, run_optimization_comparison
from benchmarks.visualization import (
    generate_time_budget_report,
    generate_optimization_comparison_chart,
)

__all__ = [
    "BenchmarkConfig",
    "BENCHMARK_CONFIGS",
    "BenchmarkRunner",
    "run_optimization_comparison",
    "create_optimization_sweep",
    "generate_time_budget_report",
    "generate_optimization_comparison_chart",
]
