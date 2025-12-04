#!/usr/bin/env python
# LongLive Comparison Video Generation
# SPDX-License-Identifier: Apache-2.0
"""
Generate side-by-side comparison videos for subjective evaluation.

Generates videos with:
- Baseline configuration
- Optimized configuration
- Aggressive configuration

Outputs individual videos + side-by-side comparison grid.

Usage:
    python scripts/generate_comparison_videos.py \
        --output-dir comparison_videos \
        --num-frames 120
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def save_video(video_tensor: torch.Tensor, path: str, fps: int = 16):
    """Save video tensor to file."""
    import imageio

    # video_tensor: [B, T, C, H, W] or [T, C, H, W]
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]

    # Convert to numpy [T, H, W, C]
    video_np = video_tensor.cpu()
    if video_np.dtype == torch.bfloat16:
        video_np = video_np.float()

    video_np = (video_np * 255).clamp(0, 255).to(torch.uint8).numpy()
    video_np = video_np.transpose(0, 2, 3, 1)  # [T, H, W, C]

    imageio.mimsave(path, video_np, fps=fps)
    print(f"  Saved: {path}")


def save_frames(video_tensor: torch.Tensor, output_dir: str, prefix: str = "frame"):
    """Save individual frames as images."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]

    video_np = video_tensor.cpu()
    if video_np.dtype == torch.bfloat16:
        video_np = video_np.float()

    # Sample frames: first, 1/4, 1/2, 3/4, last
    T = video_tensor.shape[0]
    sample_indices = [0, T//4, T//2, 3*T//4, T-1]

    for idx in sample_indices:
        frame = video_np[idx]  # [C, H, W]
        frame = (frame * 255).clamp(0, 255).to(torch.uint8)
        img = TF.to_pil_image(frame)
        img.save(output_path / f"{prefix}_{idx:04d}.png")

    print(f"  Saved {len(sample_indices)} frames to {output_dir}")


def create_comparison_grid(videos: Dict[str, torch.Tensor], output_path: str, fps: int = 16):
    """Create side-by-side comparison video."""
    import imageio

    # Get common frame count
    min_frames = min(v.shape[1] if v.dim() == 5 else v.shape[0] for v in videos.values())

    # Prepare frames
    config_names = list(videos.keys())
    num_configs = len(config_names)

    # Get frame dimensions
    sample = videos[config_names[0]]
    if sample.dim() == 5:
        sample = sample[0]
    _, H, W = sample.shape[1:]

    # Create grid frames
    grid_frames = []

    for t in range(min_frames):
        row_frames = []
        for name in config_names:
            v = videos[name]
            if v.dim() == 5:
                v = v[0]
            frame = v[t].cpu()
            if frame.dtype == torch.bfloat16:
                frame = frame.float()
            frame = (frame * 255).clamp(0, 255).to(torch.uint8).numpy()
            frame = frame.transpose(1, 2, 0)  # [H, W, C]
            row_frames.append(frame)

        # Stack horizontally
        grid_frame = np.concatenate(row_frames, axis=1)
        grid_frames.append(grid_frame)

    # Save grid video
    imageio.mimsave(output_path, grid_frames, fps=fps)
    print(f"  Saved comparison grid: {output_path}")


def generate_comparison_videos(
    config_path: str,
    output_dir: str,
    prompts: List[str],
    num_frames: int = 120,
    seed: int = 42
):
    """Generate comparison videos across configurations."""
    from utils.gpu_workflow import OptimizedInferencePipeline, OptimizationConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define configurations to compare
    configurations = {
        "baseline": OptimizationConfig(
            use_torch_compile=False,
            use_fp8=False,
            use_quantized_kv=False,
            local_attn_size=12,
        ),
        "optimized": OptimizationConfig(
            use_torch_compile=True,
            use_fp8=True,
            use_quantized_kv=True,
            local_attn_size=9,
        ),
        "aggressive": OptimizationConfig(
            use_torch_compile=True,
            use_fp8=True,
            use_quantized_kv=True,
            local_attn_size=6,
        ),
    }

    # Results storage
    all_results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
        print(f"{'='*60}")

        prompt_videos = {}
        prompt_metrics = {"prompt": prompt, "configs": {}}

        for config_name, opt_config in configurations.items():
            print(f"\n  Generating with {config_name}...")

            # Initialize pipeline
            pipeline = OptimizedInferencePipeline(config_path, opt_config)
            pipeline.setup()

            # Generate video
            import time
            start_time = time.time()
            video = pipeline.generate(prompt, num_frames=num_frames, seed=seed)
            elapsed = time.time() - start_time

            # Get metrics
            metrics = pipeline.get_metrics()
            prompt_metrics["configs"][config_name] = {
                "fps": metrics["average_fps"],
                "ms_per_frame": metrics["average_ms_per_frame"],
                "total_time": elapsed,
            }

            prompt_videos[config_name] = video

            # Save individual video
            video_dir = output_path / f"prompt_{prompt_idx:02d}"
            video_dir.mkdir(exist_ok=True)

            video_path = video_dir / f"{config_name}.mp4"
            save_video(video, str(video_path))

            # Save sample frames
            frames_dir = video_dir / f"{config_name}_frames"
            save_frames(video, str(frames_dir))

            # Cleanup
            pipeline.cleanup()
            torch.cuda.empty_cache()

        # Create comparison grid
        grid_path = video_dir / "comparison_grid.mp4"
        create_comparison_grid(prompt_videos, str(grid_path))

        all_results.append(prompt_metrics)

    # Save metrics
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_frames": num_frames,
            "seed": seed,
            "results": all_results,
        }, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")

    # Create summary HTML
    create_comparison_html(output_path, prompts, all_results)


def create_comparison_html(output_dir: Path, prompts: List[str], results: List[Dict]):
    """Create HTML page for easy comparison viewing."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>LongLive Visual Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
        h1 { text-align: center; }
        .prompt-section { margin: 40px 0; padding: 20px; background: #2a2a2a; border-radius: 8px; }
        .prompt-text { font-size: 18px; margin-bottom: 20px; color: #88c8ff; }
        .video-grid { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }
        .video-card { text-align: center; }
        .video-card video { max-width: 400px; border: 2px solid #444; border-radius: 4px; }
        .video-card h3 { margin: 10px 0 5px 0; }
        .metrics { font-size: 12px; color: #888; }
        .comparison-video { margin-top: 20px; text-align: center; }
        .comparison-video video { max-width: 100%; border: 2px solid #007acc; border-radius: 4px; }
        .frames-grid { display: flex; gap: 10px; margin-top: 20px; justify-content: center; flex-wrap: wrap; }
        .frames-grid img { max-width: 150px; border: 1px solid #444; }
    </style>
</head>
<body>
    <h1>LongLive Visual Comparison</h1>
    <p style="text-align: center; color: #888;">Baseline vs Optimized vs Aggressive</p>
"""

    for prompt_idx, (prompt, metrics) in enumerate(zip(prompts, results)):
        html += f"""
    <div class="prompt-section">
        <div class="prompt-text">Prompt {prompt_idx + 1}: "{prompt}"</div>

        <div class="video-grid">
"""
        for config_name in ["baseline", "optimized", "aggressive"]:
            config_metrics = metrics["configs"].get(config_name, {})
            fps = config_metrics.get("fps", 0)
            ms = config_metrics.get("ms_per_frame", 0)

            html += f"""
            <div class="video-card">
                <h3>{config_name.title()}</h3>
                <video controls autoplay loop muted>
                    <source src="prompt_{prompt_idx:02d}/{config_name}.mp4" type="video/mp4">
                </video>
                <div class="metrics">{fps:.1f} FPS | {ms:.1f} ms/frame</div>
            </div>
"""

        html += f"""
        </div>

        <div class="comparison-video">
            <h3>Side-by-Side Comparison</h3>
            <video controls autoplay loop muted>
                <source src="prompt_{prompt_idx:02d}/comparison_grid.mp4" type="video/mp4">
            </video>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    html_path = output_dir / "comparison.html"
    with open(html_path, 'w') as f:
        f.write(html)

    print(f"HTML comparison page saved to: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison videos")
    parser.add_argument("--config", type=str, default="configs/longlive_optimized.yaml")
    parser.add_argument("--output-dir", type=str, default="comparison_videos")
    parser.add_argument("--prompts", type=str, help="Path to prompts file")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts if not using file")
    args = parser.parse_args()

    # Load prompts
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        prompts = [
            "A cat walking gracefully on a sandy beach at sunset",
            "Ocean waves gently crashing on rocky shores",
            "A butterfly landing softly on a colorful flower",
            "Cherry blossom petals falling in slow motion",
            "A campfire burning with crackling flames under stars",
        ]

    prompts = prompts[:args.num_prompts]
    print(f"Generating comparison videos for {len(prompts)} prompts")

    generate_comparison_videos(
        config_path=args.config,
        output_dir=args.output_dir,
        prompts=prompts,
        num_frames=args.num_frames,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
