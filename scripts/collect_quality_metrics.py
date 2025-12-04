#!/usr/bin/env python
# LongLive Quality Metrics Collection
# SPDX-License-Identifier: Apache-2.0
"""
Collect quality metrics for generated videos.

Metrics:
- FVD (Frechet Video Distance) - requires reference videos
- CLIP Score - text-video alignment
- Temporal Consistency - frame-to-frame coherence

Usage:
    python scripts/collect_quality_metrics.py \
        --config configs/longlive_optimized.yaml \
        --prompts prompts.txt \
        --output-dir quality_results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class QualityMetrics:
    """Quality metrics for a single video."""
    prompt: str
    config_name: str
    clip_score: float
    temporal_consistency: float
    fvd_score: Optional[float] = None  # Requires reference

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityEvaluator:
    """
    Evaluate quality of generated videos.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.clip_model = None
        self.clip_processor = None

    def load_clip(self):
        """Load CLIP model for text-video similarity."""
        if self.clip_model is not None:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print("CLIP model loaded.")
        except ImportError:
            print("Warning: transformers not installed. CLIP score unavailable.")

    def compute_clip_score(self, video: torch.Tensor, prompt: str) -> float:
        """
        Compute CLIP score between video frames and text prompt.

        Args:
            video: Video tensor [T, C, H, W] or [T, H, W, C]
            prompt: Text prompt

        Returns:
            Average CLIP similarity score
        """
        if self.clip_model is None:
            self.load_clip()

        if self.clip_model is None:
            return 0.0

        # Sample frames (every 4th frame)
        if video.dim() == 4:
            if video.shape[-1] == 3:  # [T, H, W, C]
                video = video.permute(0, 3, 1, 2)
            # Now [T, C, H, W]

        num_frames = video.shape[0]
        sample_indices = list(range(0, num_frames, max(1, num_frames // 8)))[:8]
        sampled_frames = video[sample_indices]

        # Convert to PIL images for CLIP
        from PIL import Image
        import torchvision.transforms.functional as TF

        pil_images = [TF.to_pil_image(f) for f in sampled_frames.cpu()]

        # Compute similarity
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[prompt],
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.clip_model(**inputs)

            # Get image-text similarity
            logits_per_image = outputs.logits_per_image  # [num_images, 1]
            similarity = logits_per_image.softmax(dim=0).mean().item()

        return similarity * 100  # Scale to 0-100

    def compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """
        Compute temporal consistency score.

        Uses cosine similarity between consecutive frames.

        Args:
            video: Video tensor [T, C, H, W]

        Returns:
            Average frame-to-frame consistency score
        """
        if video.dim() == 4 and video.shape[-1] == 3:
            video = video.permute(0, 3, 1, 2)

        # Flatten frames
        T = video.shape[0]
        frames_flat = video.reshape(T, -1).float()

        # Compute cosine similarity between consecutive frames
        similarities = []
        for i in range(T - 1):
            sim = torch.nn.functional.cosine_similarity(
                frames_flat[i:i+1],
                frames_flat[i+1:i+2],
                dim=1
            )
            similarities.append(sim.item())

        return np.mean(similarities) * 100  # Scale to 0-100

    def evaluate_video(
        self,
        video: torch.Tensor,
        prompt: str,
        config_name: str
    ) -> QualityMetrics:
        """
        Evaluate a single video.

        Args:
            video: Video tensor
            prompt: Generation prompt
            config_name: Configuration name

        Returns:
            QualityMetrics object
        """
        clip_score = self.compute_clip_score(video, prompt)
        temporal_score = self.compute_temporal_consistency(video)

        return QualityMetrics(
            prompt=prompt,
            config_name=config_name,
            clip_score=clip_score,
            temporal_consistency=temporal_score,
        )


def run_quality_evaluation(
    config_path: str,
    prompts: List[str],
    output_dir: str,
    configs_to_test: Optional[List[str]] = None
):
    """
    Run quality evaluation across configurations.
    """
    from utils.gpu_workflow import OptimizedInferencePipeline, OptimizationConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    evaluator = QualityEvaluator()

    # Define configurations
    config_variations = {
        "baseline": OptimizationConfig(
            use_torch_compile=False,
            use_fp8=False,
            local_attn_size=12,
        ),
        "optimized": OptimizationConfig(
            use_torch_compile=True,
            use_fp8=True,
            local_attn_size=9,
        ),
        "aggressive": OptimizationConfig(
            use_torch_compile=True,
            use_fp8=True,
            local_attn_size=6,
        ),
    }

    if configs_to_test:
        config_variations = {k: v for k, v in config_variations.items() if k in configs_to_test}

    all_results = []

    for config_name, opt_config in config_variations.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*60}")

        # Initialize pipeline
        pipeline = OptimizedInferencePipeline(config_path, opt_config)
        pipeline.setup()

        for prompt in tqdm(prompts, desc=f"Generating ({config_name})"):
            # Generate video
            video = pipeline.generate(prompt, num_frames=60, seed=42)

            # Evaluate
            metrics = evaluator.evaluate_video(video[0], prompt, config_name)
            all_results.append(metrics)

            print(f"  Prompt: {prompt[:50]}...")
            print(f"    CLIP: {metrics.clip_score:.2f}, Temporal: {metrics.temporal_consistency:.2f}")

        pipeline.cleanup()
        torch.cuda.empty_cache()

    # Save results
    results_path = output_path / "quality_metrics.json"
    with open(results_path, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("QUALITY EVALUATION SUMMARY")
    print("="*60)

    for config_name in config_variations.keys():
        config_results = [r for r in all_results if r.config_name == config_name]
        avg_clip = np.mean([r.clip_score for r in config_results])
        avg_temporal = np.mean([r.temporal_consistency for r in config_results])
        print(f"\n{config_name}:")
        print(f"  Avg CLIP Score: {avg_clip:.2f}")
        print(f"  Avg Temporal Consistency: {avg_temporal:.2f}")

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="LongLive Quality Metrics")
    parser.add_argument("--config", type=str, default="configs/longlive_optimized.yaml")
    parser.add_argument("--prompts", type=str, help="Path to prompts file (one per line)")
    parser.add_argument("--output-dir", type=str, default="quality_results")
    parser.add_argument("--configs", type=str, nargs="+",
                        choices=["baseline", "optimized", "aggressive"],
                        help="Configurations to test")
    args = parser.parse_args()

    # Load prompts
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "A cat walking on the beach at sunset",
            "A timelapse of clouds moving across a blue sky",
            "A person playing piano in a concert hall",
            "Waves crashing on rocks by the ocean",
            "A butterfly landing on a flower",
        ]

    print(f"Testing with {len(prompts)} prompts")

    run_quality_evaluation(
        config_path=args.config,
        prompts=prompts,
        output_dir=args.output_dir,
        configs_to_test=args.configs
    )


if __name__ == "__main__":
    main()
