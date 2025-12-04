#!/usr/bin/env python
# LongLive Optimized Inference
# SPDX-License-Identifier: Apache-2.0
"""
Optimized inference script for LongLive video generation.

This script applies multiple optimizations:
- torch.compile for kernel fusion
- FP8 quantization (on H100)
- Attention window reduction
- Profiling instrumentation

Usage:
    python optimized_inference.py --config configs/longlive_optimized.yaml \
        --prompts "A cat walking on the beach" --output output.mp4
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import torch
from omegaconf import OmegaConf

from pipeline.causal_inference import CausalInferencePipeline
from utils.profiler import LongLiveProfiler, set_profiler
from utils.quantization import (
    apply_fp8_quantization,
    apply_torch_compile,
    get_model_size_mb,
    print_optimization_summary
)


def load_pipeline(args, device: torch.device) -> CausalInferencePipeline:
    """Load and optimize the pipeline."""
    print("Loading pipeline...")

    # Track original model size
    pipeline = CausalInferencePipeline(args, device=device)
    original_size = get_model_size_mb(pipeline.generator.model)

    # Apply optimizations
    opt_config = getattr(args, 'optimization', OmegaConf.create({}))

    use_compile = getattr(opt_config, 'use_torch_compile', False)
    use_fp8 = getattr(opt_config, 'use_fp8', False)
    compile_mode = getattr(opt_config, 'compile_mode', 'reduce-overhead')

    if use_torch_compile:
        print(f"Applying torch.compile (mode={compile_mode})...")
        pipeline.generator.model = apply_torch_compile(
            pipeline.generator.model,
            mode=compile_mode,
            fullgraph=False,
            dynamic=True
        )

    if use_fp8:
        print("Applying FP8 quantization...")
        pipeline.generator.model = apply_fp8_quantization(pipeline.generator.model)

    # Move to device
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    print_optimization_summary(
        pipeline.generator.model,
        original_size,
        use_compile=use_compile,
        use_fp8=use_fp8
    )

    return pipeline


def run_inference(
    pipeline: CausalInferencePipeline,
    prompts: List[str],
    args,
    device: torch.device,
    profiler: Optional[LongLiveProfiler] = None
) -> torch.Tensor:
    """Run optimized inference with profiling."""
    num_frames = args.num_output_frames

    # Generate noise
    torch.manual_seed(args.seed)
    noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # Profile inference
    if profiler:
        profiler.reset()
        profiler.snapshot_memory("pre_inference")

    start_time = time.perf_counter()

    with torch.no_grad():
        if profiler:
            with profiler.profile_region("inference"):
                video = pipeline.inference(
                    noise=noise,
                    text_prompts=prompts,
                    return_latents=False,
                    profile=False  # Use our profiler
                )
        else:
            video = pipeline.inference(
                noise=noise,
                text_prompts=prompts,
                return_latents=False,
                profile=True  # Use built-in profiler
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    if profiler:
        profiler.snapshot_memory("post_inference")

    # Print timing
    fps = num_frames / elapsed
    ms_per_frame = elapsed * 1000 / num_frames

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Frames: {num_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"ms/frame: {ms_per_frame:.2f}")
    print(f"Target (40ms): {'PASS' if ms_per_frame <= 40 else 'FAIL'}")
    print("=" * 60)

    return video


def save_video(video: torch.Tensor, output_path: str, fps: int = 16):
    """Save video tensor to file."""
    import imageio

    # video: [B, T, C, H, W] in [0, 1]
    video = video[0]  # Remove batch dim
    video = (video * 255).to(torch.uint8)
    video = video.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]

    print(f"Saving video to {output_path}...")
    imageio.mimwrite(output_path, video, fps=fps)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LongLive Optimized Inference")

    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--prompts", type=str, nargs="+", required=True, help="Text prompts")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num-frames", type=int, help="Override number of frames")
    parser.add_argument("--seed", type=int, help="Override random seed")

    # Optimization overrides
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 quantization")
    parser.add_argument("--window-size", type=int, help="Override attention window size")

    # Profiling
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    parser.add_argument("--profile-output", type=str, help="Profile output JSON path")

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Apply overrides
    if args.num_frames:
        config.num_output_frames = args.num_frames
    if args.seed is not None:
        config.seed = args.seed
    if args.window_size:
        config.model_kwargs.local_attn_size = args.window_size

    # Handle optimization overrides
    if not hasattr(config, 'optimization'):
        config.optimization = OmegaConf.create({})
    if args.no_compile:
        config.optimization.use_torch_compile = False
    if args.no_fp8:
        config.optimization.use_fp8 = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Setup profiler
    profiler = None
    if args.profile:
        profiler = LongLiveProfiler(enabled=True, sync_cuda=True)
        set_profiler(profiler)

    # Load pipeline
    pipeline = load_pipeline(config, device)

    # Run inference
    video = run_inference(pipeline, args.prompts, config, device, profiler)

    # Save video
    save_video(video, args.output)

    # Save profiling results
    if profiler and args.profile_output:
        profiler.export_json(args.profile_output)
        print(f"Profile saved to: {args.profile_output}")

    if profiler:
        print("\n" + profiler.generate_report())


if __name__ == "__main__":
    main()
