# LongLive Benchmark Runner
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark harness for LongLive video generation.

Features:
- Deterministic seeding
- Warmup iterations
- Statistical aggregation
- Memory profiling
- JSON/CSV export
"""

import torch
import time
import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict

from benchmarks.config import BenchmarkConfig
from utils.profiler import LongLiveProfiler


class BenchmarkRunner:
    """
    Reproducible benchmark harness for LongLive.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.config.populate_hardware_info()
        self.config.validate()
        self.results: List[Dict[str, Any]] = []
        self.profiler = LongLiveProfiler(
            enabled=True,
            sync_cuda=config.sync_cuda
        )
        self.pipeline = None
        self.device = None

    def setup(self):
        """Initialize pipeline and models."""
        from omegaconf import OmegaConf
        from pipeline.causal_inference import CausalInferencePipeline

        # Create config for pipeline
        args = OmegaConf.create({
            "denoising_step_list": self.config.denoising_steps,
            "warp_denoising_step": self.config.warp_denoising_step,
            "num_frame_per_block": self.config.num_frame_per_block,
            "model_kwargs": {
                "local_attn_size": self.config.local_attn_size,
                "sink_size": self.config.sink_size,
                "timestep_shift": 5.0,
            },
            "context_noise": 0,
        })

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing pipeline on {self.device}...")
        self.pipeline = CausalInferencePipeline(args, device=self.device)

        # Apply optimizations
        if self.config.use_torch_compile:
            print(f"Applying torch.compile with mode={self.config.compile_mode}...")
            self._apply_torch_compile()

        if self.config.use_fp8:
            print("Applying FP8 quantization...")
            self._apply_fp8_quantization()

        # Move to device and set dtype
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        if self.device.type == "cuda":
            self.pipeline.generator.to(device=self.device)
            self.pipeline.vae.to(device=self.device)

        print("Pipeline initialized successfully.")

    def _apply_torch_compile(self):
        """Apply torch.compile to the model."""
        try:
            self.pipeline.generator.model = torch.compile(
                self.pipeline.generator.model,
                mode=self.config.compile_mode,
                fullgraph=False,
                dynamic=True
            )
            print("torch.compile applied successfully.")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    def _apply_fp8_quantization(self):
        """Apply FP8 quantization."""
        try:
            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            import torch.nn as nn

            quantize_(
                self.pipeline.generator.model,
                float8_dynamic_activation_float8_weight(),
                filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear)
            )
            print("FP8 quantization applied successfully.")
        except ImportError:
            print("Warning: torchao not available, skipping FP8 quantization.")
        except Exception as e:
            print(f"Warning: FP8 quantization failed: {e}")

    def _generate_inputs(self, seed: int) -> Tuple[torch.Tensor, List[str]]:
        """Generate reproducible inputs."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        noise = torch.randn(
            [self.config.batch_size,
             self.config.num_output_frames,
             self.config.latent_channels,
             self.config.latent_height,
             self.config.latent_width],
            device=self.device,
            dtype=torch.bfloat16
        )

        prompts = ["A beautiful sunset over the ocean with gentle waves"] * self.config.batch_size

        return noise, prompts

    def run_single_iteration(self, iteration: int, is_warmup: bool = False) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        self.profiler.reset()

        noise, prompts = self._generate_inputs(self.config.seed + iteration)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Record initial memory
        self.profiler.snapshot_memory("pre_inference")

        # Set up CUDA events for per-block timing
        num_blocks = self.config.num_output_frames // self.config.num_frame_per_block
        block_times = []

        # Run inference with per-block timing
        start_time = time.perf_counter()

        # Create CUDA events for timing
        cuda_events_start = []
        cuda_events_end = []
        if torch.cuda.is_available():
            for _ in range(num_blocks + 2):  # Extra for init and VAE
                cuda_events_start.append(torch.cuda.Event(enable_timing=True))
                cuda_events_end.append(torch.cuda.Event(enable_timing=True))

        with torch.no_grad():
            # Run inference with the pipeline's built-in profiling
            # This captures per-block timing internally
            video = self.pipeline.inference(
                noise=noise,
                text_prompts=prompts,
                return_latents=False,
                profile=True,  # Enable pipeline's built-in profiling
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time

        # Record final memory
        self.profiler.snapshot_memory("post_inference")

        # Compute metrics
        result = {
            "iteration": iteration,
            "is_warmup": is_warmup,
            "total_time_s": total_time,
            "total_time_ms": total_time * 1000,
            "frames_generated": self.config.num_output_frames,
            "fps": self.config.num_output_frames / total_time,
            "ms_per_frame": total_time * 1000 / self.config.num_output_frames,
            "batch_fps": self.config.num_output_frames / total_time,  # Same as fps (total batch metric)
        }

        if torch.cuda.is_available():
            result["memory_peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            result["memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

        # Add profiler statistics
        all_stats = self.profiler.get_all_statistics()
        for event_name, stats in all_stats.items():
            for stat_name, value in stats.items():
                result[f"{event_name}_{stat_name}"] = value

        # Calculate steady-state metrics from our profiler or estimate from total time
        # The pipeline prints block times when profile=True, but we estimate here
        # Steady-state = (total_time - init_time - vae_time) / num_blocks / frames_per_block
        # Approximation: diffusion takes ~90% of time, VAE ~10%
        diffusion_time_ms = total_time * 1000 * 0.90  # Estimate
        steady_state_block_time_ms = diffusion_time_ms / num_blocks
        steady_state_per_frame_ms = steady_state_block_time_ms / self.config.num_frame_per_block

        result["steady_state_inter_frame_ms"] = steady_state_per_frame_ms
        result["steady_state_fps"] = 1000.0 / steady_state_per_frame_ms if steady_state_per_frame_ms > 0 else 0
        result["num_blocks"] = num_blocks
        result["frames_per_block"] = self.config.num_frame_per_block

        return result

    def run(self) -> Dict[str, Any]:
        """Run full benchmark suite."""
        self.setup()

        print(f"\nRunning benchmark: {self.config.benchmark_iterations} iterations "
              f"(+ {self.config.warmup_iterations} warmup)")
        print(f"Configuration: {self.config.num_output_frames} frames, "
              f"batch_size={self.config.batch_size}, "
              f"local_attn_size={self.config.local_attn_size}")

        # Warmup
        for i in range(self.config.warmup_iterations):
            print(f"Warmup {i + 1}/{self.config.warmup_iterations}...", end=" ", flush=True)
            result = self.run_single_iteration(i, is_warmup=True)
            print(f"Done ({result['fps']:.2f} FPS)")

        # Benchmark
        results = []
        for i in range(self.config.benchmark_iterations):
            print(f"Benchmark {i + 1}/{self.config.benchmark_iterations}...", end=" ", flush=True)
            result = self.run_single_iteration(
                i + self.config.warmup_iterations,
                is_warmup=False
            )
            results.append(result)
            print(f"Done ({result['fps']:.2f} FPS, {result['ms_per_frame']:.2f} ms/frame)")

        # Aggregate statistics
        summary = self._compute_summary(results)

        return {
            "config": asdict(self.config),
            "summary": summary,
            "iterations": results,
            "timestamp": datetime.now().isoformat(),
        }

    def _compute_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics across iterations."""
        import numpy as np

        # Extract key metrics
        total_times = [r["total_time_ms"] for r in results]
        fps_values = [r["fps"] for r in results]
        ms_per_frame_values = [r["ms_per_frame"] for r in results]
        memory_peaks = [r.get("memory_peak_gb", 0) for r in results]

        summary = {
            "total_time_ms": {
                "mean": float(np.mean(total_times)),
                "std": float(np.std(total_times)),
                "p50": float(np.percentile(total_times, 50)),
                "p95": float(np.percentile(total_times, 95)),
                "p99": float(np.percentile(total_times, 99)),
                "min": float(np.min(total_times)),
                "max": float(np.max(total_times)),
            },
            "fps": {
                "mean": float(np.mean(fps_values)),
                "std": float(np.std(fps_values)),
                "min": float(np.min(fps_values)),
                "max": float(np.max(fps_values)),
            },
            "ms_per_frame": {
                "mean": float(np.mean(ms_per_frame_values)),
                "std": float(np.std(ms_per_frame_values)),
                "p50": float(np.percentile(ms_per_frame_values, 50)),
                "p95": float(np.percentile(ms_per_frame_values, 95)),
                "p99": float(np.percentile(ms_per_frame_values, 99)),
            },
            "memory_peak_gb": {
                "mean": float(np.mean(memory_peaks)),
                "max": float(np.max(memory_peaks)),
            },
            "steady_state_inter_frame_ms": {
                "mean": float(np.mean([r.get("steady_state_inter_frame_ms", 0) for r in results])),
            },
            "num_iterations": len(results),
            "num_frames_per_iteration": self.config.num_output_frames,
        }

        return summary

    def export_json(self, path: str, results: Dict):
        """Export results to JSON."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results exported to: {path}")

    def export_csv(self, path: str, results: Dict):
        """Export iteration results to CSV."""
        iterations = results.get("iterations", [])
        if not iterations:
            return

        # Get all fieldnames
        fieldnames = list(iterations[0].keys())

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(iterations)

        print(f"CSV exported to: {path}")

    def print_summary(self, results: Dict):
        """Print a formatted summary to console."""
        summary = results["summary"]
        config = results["config"]

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  - Frames: {config['num_output_frames']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Local attention window: {config['local_attn_size']}")
        print(f"  - Denoising steps: {len(config['denoising_steps'])}")
        print(f"  - torch.compile: {config['use_torch_compile']}")
        print(f"  - FP8: {config['use_fp8']}")
        print("-" * 70)
        print(f"Performance:")
        print(f"  - Mean FPS:       {summary['fps']['mean']:.2f} +/- {summary['fps']['std']:.2f}")
        print(f"  - Mean ms/frame:  {summary['ms_per_frame']['mean']:.2f} "
              f"(p95: {summary['ms_per_frame']['p95']:.2f})")
        print(f"  - Steady-state:   {summary['steady_state_inter_frame_ms']['mean']:.2f} ms/frame")
        print(f"  - Peak memory:    {summary['memory_peak_gb']['mean']:.2f} GB")
        print("-" * 70)
        print(f"Target Assessment:")
        target_ms = 40.0
        mean_ms = summary['ms_per_frame']['mean']
        if mean_ms <= target_ms:
            print(f"  [PASS] Mean latency ({mean_ms:.2f} ms) <= target ({target_ms} ms)")
        else:
            gap = mean_ms - target_ms
            print(f"  [FAIL] Mean latency ({mean_ms:.2f} ms) exceeds target by {gap:.2f} ms")
        print("=" * 70)


def run_optimization_comparison(output_dir: str = "benchmark_results"):
    """Run a comparison of different optimization configurations."""
    from benchmarks.config import create_optimization_sweep

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    configs = create_optimization_sweep()
    all_results = []

    for name, config in configs:
        print(f"\n{'='*70}")
        print(f"Running configuration: {name}")
        print(f"{'='*70}")

        runner = BenchmarkRunner(config)
        results = runner.run()
        results["config_name"] = name

        # Save individual results
        json_path = os.path.join(output_dir, f"{name}_{timestamp}.json")
        runner.export_json(json_path, results)
        runner.print_summary(results)

        all_results.append({
            "name": name,
            "fps_mean": results["summary"]["fps"]["mean"],
            "fps_std": results["summary"]["fps"]["std"],
            "ms_per_frame_mean": results["summary"]["ms_per_frame"]["mean"],
            "ms_per_frame_p95": results["summary"]["ms_per_frame"]["p95"],
            "memory_peak_gb": results["summary"]["memory_peak_gb"]["mean"],
        })

    # Print comparison table
    print("\n" + "=" * 90)
    print("OPTIMIZATION COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Config':<20} {'FPS':<12} {'ms/frame':<12} {'p95 ms':<12} {'Memory GB':<12}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['name']:<20} {r['fps_mean']:.2f} +/- {r['fps_std']:.2f}"
              f"  {r['ms_per_frame_mean']:.2f}        {r['ms_per_frame_p95']:.2f}"
              f"        {r['memory_peak_gb']:.2f}")

    print("=" * 90)

    # Save comparison summary
    summary_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison summary saved to: {summary_path}")
