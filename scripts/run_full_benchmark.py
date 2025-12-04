#!/usr/bin/env python
# LongLive Full Benchmark Suite
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive benchmark suite for LongLive on H100.

Runs all optimization configurations and collects:
- Latency metrics (steady-state, cross-batch, prompt-switch)
- Memory usage
- Quality metrics (FVD, CLIP score approximation)

Usage:
    python scripts/run_full_benchmark.py [--output-dir results] [--quick]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import BenchmarkConfig
from benchmarks.visualization import generate_time_budget_report, generate_optimization_comparison_chart
from utils.profiler import LongLiveProfiler
from utils.job_callbacks import JobReporter, create_reporter


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_name: str
    fps_mean: float
    fps_std: float
    ms_per_frame_mean: float
    ms_per_frame_p50: float
    ms_per_frame_p95: float
    ms_per_frame_p99: float
    steady_state_ms: float
    memory_peak_gb: float
    memory_allocated_gb: float
    total_time_s: float
    num_frames: int
    num_iterations: int

    # Quality metrics (if computed)
    fvd_score: Optional[float] = None
    clip_score: Optional[float] = None
    temporal_consistency: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FullBenchmarkRunner:
    """
    Run comprehensive benchmarks across all optimization configurations.
    """

    def __init__(
        self,
        output_dir: str = "benchmark_results",
        config_path: str = "configs/longlive_optimized.yaml",
        quick: bool = False,
        slack_webhook: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path
        self.quick = quick

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[BenchmarkResult] = []

        # Initialize job reporter for progress tracking
        self.reporter = JobReporter(
            webhook_url=slack_webhook or os.environ.get("SLACK_WEBHOOK_URL"),
            webhook_format="slack",
            progress_file="/tmp/benchmark_progress.json",
            log_file=str(self.output_dir / f"benchmark_{self.timestamp}.log"),
        )

        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This benchmark requires GPU.")

        self.device = torch.device("cuda")
        self.gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {self.gpu_name}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")

    def get_configurations(self) -> List[Dict[str, Any]]:
        """Define all benchmark configurations."""

        base_frames = 24 if self.quick else 120
        iterations = 3 if self.quick else 10
        warmup = 1 if self.quick else 2

        configs = [
            # Baseline
            {
                "name": "baseline",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": False,
                "use_fp8": False,
                "local_attn_size": 12,
            },
            # torch.compile only
            {
                "name": "torch_compile",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": True,
                "compile_mode": "reduce-overhead",
                "use_fp8": False,
                "local_attn_size": 12,
            },
            # Window size variations
            {
                "name": "window_9",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": False,
                "use_fp8": False,
                "local_attn_size": 9,
            },
            {
                "name": "window_6",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": False,
                "use_fp8": False,
                "local_attn_size": 6,
            },
            # FP8 quantization
            {
                "name": "fp8",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": False,
                "use_fp8": True,
                "local_attn_size": 12,
            },
            # Combined optimizations
            {
                "name": "compile_window9",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": True,
                "compile_mode": "reduce-overhead",
                "use_fp8": False,
                "local_attn_size": 9,
            },
            {
                "name": "compile_fp8",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": True,
                "compile_mode": "reduce-overhead",
                "use_fp8": True,
                "local_attn_size": 12,
            },
            # Full optimization (sweet spot)
            {
                "name": "full_optimized",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": True,
                "compile_mode": "reduce-overhead",
                "use_fp8": True,
                "local_attn_size": 9,
            },
            # Aggressive optimization (quality tradeoff)
            {
                "name": "aggressive",
                "num_output_frames": base_frames,
                "benchmark_iterations": iterations,
                "warmup_iterations": warmup,
                "use_torch_compile": True,
                "compile_mode": "max-autotune",
                "use_fp8": True,
                "local_attn_size": 6,
            },
        ]

        return configs

    def run_single_config(self, config: Dict[str, Any], config_idx: int, total_configs: int) -> BenchmarkResult:
        """Run benchmark for a single configuration."""

        config_name = config.pop("name")
        print(f"\n{'='*60}")
        print(f"Running: {config_name}")
        print(f"{'='*60}")

        # Update progress
        progress = 10 + (80 * config_idx / total_configs)
        self.reporter.update_progress(
            stage="benchmarking",
            progress=progress,
            task=f"Running {config_name}",
            tasks_completed=config_idx,
        )
        self.reporter.log(f"Starting benchmark: {config_name}")

        # Create benchmark config
        benchmark_config = BenchmarkConfig(**config)
        benchmark_config.populate_hardware_info()

        # Import here to avoid loading model until needed
        from benchmarks.runner import BenchmarkRunner

        runner = BenchmarkRunner(benchmark_config)
        results = runner.run()

        # Extract metrics
        summary = results["summary"]

        result = BenchmarkResult(
            config_name=config_name,
            fps_mean=summary["fps"]["mean"],
            fps_std=summary["fps"]["std"],
            ms_per_frame_mean=summary["ms_per_frame"]["mean"],
            ms_per_frame_p50=summary["ms_per_frame"]["p50"],
            ms_per_frame_p95=summary["ms_per_frame"]["p95"],
            ms_per_frame_p99=summary["ms_per_frame"]["p99"],
            steady_state_ms=summary["steady_state_inter_frame_ms"]["mean"],
            memory_peak_gb=summary["memory_peak_gb"]["mean"],
            memory_allocated_gb=summary["memory_peak_gb"].get("max", summary["memory_peak_gb"]["mean"]),
            total_time_s=summary["total_time_ms"]["mean"] / 1000,
            num_frames=benchmark_config.num_output_frames,
            num_iterations=summary["num_iterations"],
        )

        # Save individual result
        result_path = self.output_dir / f"{config_name}_{self.timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump({
                "config": results["config"],
                "summary": summary,
                "result": result.to_dict(),
            }, f, indent=2, default=str)

        print(f"\nResults for {config_name}:")
        print(f"  FPS: {result.fps_mean:.2f} +/- {result.fps_std:.2f}")
        print(f"  ms/frame: {result.ms_per_frame_mean:.2f} (p95: {result.ms_per_frame_p95:.2f})")
        print(f"  Steady-state: {result.steady_state_ms:.2f} ms")
        print(f"  Memory: {result.memory_peak_gb:.2f} GB")

        # Clear GPU memory
        del runner
        torch.cuda.empty_cache()

        return result

    def run_all(self):
        """Run all benchmark configurations."""

        print("="*60)
        print("LONGLIVE FULL BENCHMARK SUITE")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Quick mode: {self.quick}")
        print(f"Timestamp: {self.timestamp}")

        configs = self.get_configurations()
        print(f"\nConfigurations to test: {len(configs)}")

        # Start job reporting
        self.reporter.start_job("benchmark", total_tasks=len(configs))
        self.reporter.update_progress(stage="setup", progress=5, task="Loading configurations")

        for idx, config in enumerate(configs):
            try:
                result = self.run_single_config(config.copy(), idx, len(configs))
                self.results.append(result)

                # Report result
                self.reporter.update_progress(
                    fps=result.fps_mean,
                    latency_ms=result.ms_per_frame_mean,
                    memory_gb=result.memory_peak_gb,
                    metrics={config.get("name", f"config_{idx}"): {
                        "fps": result.fps_mean,
                        "latency_ms": result.ms_per_frame_mean,
                    }}
                )
                self.reporter.log(f"Completed {result.config_name}: {result.fps_mean:.1f} FPS, {result.ms_per_frame_mean:.1f} ms")

            except Exception as e:
                print(f"ERROR running {config.get('name', 'unknown')}: {e}")
                self.reporter.error(f"Failed {config.get('name', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()

        # Generate summary
        self.reporter.update_progress(stage="generating_report", progress=95, task="Generating summary")
        self.generate_summary()

        # Complete job
        best_fps = max(self.results, key=lambda x: x.fps_mean) if self.results else None
        self.reporter.complete_job(summary={
            "best_config": best_fps.config_name if best_fps else None,
            "best_fps": best_fps.fps_mean if best_fps else 0,
            "total_configs": len(self.results),
        })

    def generate_summary(self):
        """Generate summary reports."""

        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Print comparison table
        print(f"\n{'Config':<20} {'FPS':>10} {'ms/frame':>12} {'p95 ms':>10} {'Memory GB':>12} {'Target':>10}")
        print("-"*80)

        target_ms = 40.0

        for r in self.results:
            meets_target = "✓" if r.ms_per_frame_mean <= target_ms else "✗"
            print(f"{r.config_name:<20} {r.fps_mean:>10.2f} {r.ms_per_frame_mean:>12.2f} "
                  f"{r.ms_per_frame_p95:>10.2f} {r.memory_peak_gb:>12.2f} {meets_target:>10}")

        print("-"*80)

        # Find best configuration
        best_fps = max(self.results, key=lambda x: x.fps_mean)
        best_latency = min(self.results, key=lambda x: x.ms_per_frame_mean)

        print(f"\nBest FPS: {best_fps.config_name} ({best_fps.fps_mean:.2f} FPS)")
        print(f"Best Latency: {best_latency.config_name} ({best_latency.ms_per_frame_mean:.2f} ms)")

        # Calculate speedup
        baseline = next((r for r in self.results if r.config_name == "baseline"), None)
        if baseline:
            print(f"\nSpeedup vs Baseline:")
            for r in self.results:
                if r.config_name != "baseline":
                    speedup = baseline.ms_per_frame_mean / r.ms_per_frame_mean
                    print(f"  {r.config_name}: {speedup:.2f}x")

        # Save summary JSON
        summary_path = self.output_dir / f"summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "gpu": self.gpu_name,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "results": [r.to_dict() for r in self.results],
                "best_fps": best_fps.config_name,
                "best_latency": best_latency.config_name,
            }, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")

        # Generate HTML comparison chart
        chart_path = self.output_dir / f"comparison_{self.timestamp}.html"
        generate_optimization_comparison_chart(
            [r.to_dict() for r in self.results],
            str(chart_path)
        )
        print(f"Comparison chart saved to: {chart_path}")

        # Generate CSV for easy analysis
        csv_path = self.output_dir / f"results_{self.timestamp}.csv"
        with open(csv_path, 'w') as f:
            headers = list(self.results[0].to_dict().keys())
            f.write(",".join(headers) + "\n")
            for r in self.results:
                values = [str(v) if v is not None else "" for v in r.to_dict().values()]
                f.write(",".join(values) + "\n")
        print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="LongLive Full Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--config", type=str, default="configs/longlive_optimized.yaml",
                        help="Base config file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer frames/iterations)")
    parser.add_argument("--slack-webhook", type=str,
                        help="Slack webhook URL for progress notifications")
    args = parser.parse_args()

    runner = FullBenchmarkRunner(
        output_dir=args.output_dir,
        config_path=args.config,
        quick=args.quick,
        slack_webhook=args.slack_webhook,
    )

    start_time = time.time()
    runner.run_all()
    elapsed = time.time() - start_time

    print(f"\nTotal benchmark time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
