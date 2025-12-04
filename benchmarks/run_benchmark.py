#!/usr/bin/env python
# LongLive Benchmark CLI
# SPDX-License-Identifier: Apache-2.0
"""
CLI for running LongLive benchmarks.

Usage:
    python benchmarks/run_benchmark.py --config standard --output results/
    python benchmarks/run_benchmark.py --config-file custom_config.yaml
    python benchmarks/run_benchmark.py --comparison  # Run optimization comparison
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.config import BenchmarkConfig, BENCHMARK_CONFIGS
from benchmarks.runner import BenchmarkRunner, run_optimization_comparison


def main():
    parser = argparse.ArgumentParser(
        description="LongLive Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run standard benchmark
    python benchmarks/run_benchmark.py --config standard

    # Run quick benchmark for testing
    python benchmarks/run_benchmark.py --config quick

    # Run with custom settings
    python benchmarks/run_benchmark.py --frames 60 --iterations 5 --window-size 9

    # Run optimization comparison sweep
    python benchmarks/run_benchmark.py --comparison

    # Run with torch.compile
    python benchmarks/run_benchmark.py --config standard --compile

    # Run with FP8 quantization
    python benchmarks/run_benchmark.py --config standard --fp8
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=list(BENCHMARK_CONFIGS.keys()),
        default="standard",
        help="Predefined config name"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Custom config YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run full optimization comparison sweep"
    )

    # Override options
    parser.add_argument("--iterations", type=int, help="Override benchmark iterations")
    parser.add_argument("--warmup", type=int, help="Override warmup iterations")
    parser.add_argument("--frames", type=int, help="Override num_output_frames")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--window-size", type=int, help="Override local_attn_size")
    parser.add_argument("--sink-size", type=int, help="Override sink_size")

    # Optimization flags
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["reduce-overhead", "max-autotune", "default"],
        help="torch.compile mode"
    )
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 quantization")
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA graph capture")

    # Output options
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Handle comparison mode
    if args.comparison:
        print("Running optimization comparison sweep...")
        run_optimization_comparison(args.output)
        return

    # Load config
    if args.config_file:
        config = BenchmarkConfig.from_yaml(args.config_file)
    else:
        config = BENCHMARK_CONFIGS[args.config]

    # Apply overrides
    if args.iterations is not None:
        config.benchmark_iterations = args.iterations
    if args.warmup is not None:
        config.warmup_iterations = args.warmup
    if args.frames is not None:
        config.num_output_frames = args.frames
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.window_size is not None:
        config.local_attn_size = args.window_size
    if args.sink_size is not None:
        config.sink_size = args.sink_size

    # Apply optimization flags
    if args.compile:
        config.use_torch_compile = True
        config.compile_mode = args.compile_mode
    if args.fp8:
        config.use_fp8 = True
    if args.cuda_graph:
        config.use_cuda_graph = True

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Export results
    config_name = args.config_file.replace("/", "_").replace(".yaml", "") if args.config_file else args.config
    json_path = os.path.join(args.output, f"benchmark_{config_name}_{timestamp}.json")
    runner.export_json(json_path, results)

    if not args.no_csv:
        csv_path = os.path.join(args.output, f"benchmark_{config_name}_{timestamp}.csv")
        runner.export_csv(csv_path, results)

    # Print summary
    if not args.quiet:
        runner.print_summary(results)


if __name__ == "__main__":
    main()
