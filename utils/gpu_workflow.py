# LongLive GPU Workflow Integration
# SPDX-License-Identifier: Apache-2.0
"""
GPU-optimized workflow that integrates all advanced optimizations.

This module provides a unified interface for running LongLive inference
with all optimizations enabled for maximum throughput on H100/A100 GPUs.

Usage:
    python -m utils.gpu_workflow --config configs/longlive_optimized.yaml

Or programmatically:
    from utils.gpu_workflow import OptimizedInferencePipeline
    pipeline = OptimizedInferencePipeline(config_path)
    pipeline.setup()
    video = pipeline.generate(prompt)
"""

import torch
import torch.nn as nn
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse

from utils.profiler import LongLiveProfiler, set_profiler
from utils.quantization import (
    apply_torch_compile,
    apply_fp8_quantization,
    CUDAGraphWrapper,
    get_model_size_mb,
    print_optimization_summary,
)
from utils.advanced_optimizations import (
    QuantizedKVCache,
    StreamingVAEDecoder,
    PrefixCache,
    AsyncNoisePrefetcher,
    TripleBuffer,
    apply_all_optimizations,
)


@dataclass
class OptimizationConfig:
    """Configuration for GPU optimizations."""
    # Model optimizations
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "reduce-overhead", "max-autotune"
    use_fp8: bool = True
    use_cuda_graph: bool = False  # Experimental

    # KV cache optimizations
    use_quantized_kv: bool = True
    kv_quantization: str = "int8"  # "int8" or "fp8"

    # VAE optimizations
    use_streaming_vae: bool = True

    # Memory optimizations
    use_prefix_cache: bool = True
    use_async_noise: bool = True
    use_triple_buffer: bool = False  # For streaming

    # Attention optimizations
    local_attn_size: int = 9  # Reduced from 12
    sink_size: int = 3

    # Inference settings
    num_frame_per_block: int = 3
    denoising_steps: List[int] = None

    def __post_init__(self):
        if self.denoising_steps is None:
            self.denoising_steps = [1000, 750, 500, 250]


class OptimizedInferencePipeline:
    """
    Fully optimized inference pipeline for LongLive.

    Integrates all optimization techniques:
    - torch.compile for kernel fusion
    - FP8/INT8 quantization
    - Quantized KV cache
    - Streaming VAE decoder
    - Async noise prefetching
    - Prefix caching

    Example:
        pipeline = OptimizedInferencePipeline("configs/longlive_optimized.yaml")
        pipeline.setup()
        video = pipeline.generate("A cat walking on the beach", num_frames=120)
    """

    def __init__(
        self,
        config_path: str,
        optimization_config: Optional[OptimizationConfig] = None
    ):
        self.config_path = config_path
        self.opt_config = optimization_config or OptimizationConfig()

        self.config = None
        self.pipeline = None
        self.device = None
        self.dtype = torch.bfloat16

        # Optimization components
        self.kv_cache: Optional[QuantizedKVCache] = None
        self.streaming_vae: Optional[StreamingVAEDecoder] = None
        self.prefix_cache: Optional[PrefixCache] = None
        self.noise_prefetcher: Optional[AsyncNoisePrefetcher] = None
        self.triple_buffer: Optional[TripleBuffer] = None

        # Profiling
        self.profiler = LongLiveProfiler(enabled=True)
        set_profiler(self.profiler)

        # Metrics
        self._total_frames = 0
        self._total_time = 0.0
        self._warmup_complete = False

    def setup(self):
        """Initialize pipeline with all optimizations."""
        print("=" * 60)
        print("INITIALIZING OPTIMIZED LONGLIVE PIPELINE")
        print("=" * 60)

        self.config = OmegaConf.load(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type != "cuda":
            print("WARNING: Running on CPU. GPU optimizations will be disabled.")

        self._load_pipeline()
        self._apply_optimizations()
        self._setup_components()
        self._warmup()

        print("\nPipeline ready for inference!")

    def _load_pipeline(self):
        """Load the base LongLive pipeline."""
        print("\n[1/4] Loading base pipeline...")

        from pipeline.causal_inference import CausalInferencePipeline

        # Override config with optimized settings
        self.config.model_kwargs.local_attn_size = self.opt_config.local_attn_size
        self.config.model_kwargs.sink_size = self.opt_config.sink_size
        self.config.denoising_step_list = self.opt_config.denoising_steps
        self.config.num_frame_per_block = self.opt_config.num_frame_per_block

        self.pipeline = CausalInferencePipeline(self.config, device=self.device)
        self.pipeline = self.pipeline.to(dtype=self.dtype)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)

        original_size = get_model_size_mb(self.pipeline.generator.model)
        print(f"   Model loaded: {original_size:.1f} MB")

    def _apply_optimizations(self):
        """Apply model-level optimizations."""
        print("\n[2/4] Applying optimizations...")

        original_size = get_model_size_mb(self.pipeline.generator.model)

        # torch.compile
        if self.opt_config.use_torch_compile and self.device.type == "cuda":
            print(f"   - Applying torch.compile (mode={self.opt_config.compile_mode})")
            self.pipeline.generator.model = apply_torch_compile(
                self.pipeline.generator.model,
                mode=self.opt_config.compile_mode
            )

        # FP8 quantization
        if self.opt_config.use_fp8 and self.device.type == "cuda":
            print("   - Applying FP8 quantization")
            self.pipeline.generator.model = apply_fp8_quantization(
                self.pipeline.generator.model
            )

        print_optimization_summary(
            self.pipeline.generator.model,
            original_size_mb=original_size,
            use_compile=self.opt_config.use_torch_compile,
            use_fp8=self.opt_config.use_fp8,
            use_cuda_graph=self.opt_config.use_cuda_graph,
        )

    def _setup_components(self):
        """Setup advanced optimization components."""
        print("\n[3/4] Setting up optimization components...")

        # Quantized KV cache
        if self.opt_config.use_quantized_kv:
            print(f"   - Quantized KV cache ({self.opt_config.kv_quantization})")
            self.kv_cache = QuantizedKVCache(
                num_layers=getattr(self.config.model_kwargs, 'num_transformer_blocks', 30),
                max_seq_len=self.opt_config.local_attn_size * 1560,  # frame_seq_length
                num_heads=12,
                head_dim=128,
                device=self.device,
                quantization=self.opt_config.kv_quantization
            )
            print(f"     Memory usage: {self.kv_cache.memory_usage_mb():.1f} MB")

        # Streaming VAE decoder
        if self.opt_config.use_streaming_vae and self.device.type == "cuda":
            print("   - Streaming VAE decoder")
            self.streaming_vae = StreamingVAEDecoder(
                self.pipeline.vae,
                device=self.device
            )

        # Prefix cache
        if self.opt_config.use_prefix_cache:
            print("   - Prefix cache (50 entries)")
            self.prefix_cache = PrefixCache(max_cache_size=50)

        # Async noise prefetcher
        if self.opt_config.use_async_noise and self.device.type == "cuda":
            print("   - Async noise prefetcher")
            self.noise_prefetcher = AsyncNoisePrefetcher(self.device)

        # Triple buffer for streaming
        if self.opt_config.use_triple_buffer and self.device.type == "cuda":
            print("   - Triple buffer for streaming")
            buffer_shape = (
                1,
                self.opt_config.num_frame_per_block,
                16,
                60,
                104
            )
            self.triple_buffer = TripleBuffer(
                buffer_shape=buffer_shape,
                dtype=self.dtype,
                device=self.device
            )

    def _warmup(self, num_warmup_frames: int = 6):
        """Run warmup inference to trigger JIT compilation."""
        print(f"\n[4/4] Running warmup ({num_warmup_frames} frames)...")

        torch.manual_seed(0)
        noise = torch.randn(
            [1, num_warmup_frames, 16, 60, 104],
            device=self.device,
            dtype=self.dtype
        )

        start_time = time.time()

        with torch.no_grad():
            _ = self.pipeline.inference(
                noise=noise,
                text_prompts=["Warmup inference for JIT compilation"],
                return_latents=False,
                profile=False
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        warmup_time = time.time() - start_time
        print(f"   Warmup complete: {warmup_time:.2f}s")
        self._warmup_complete = True

    def generate(
        self,
        prompt: str,
        num_frames: int = 120,
        seed: Optional[int] = None,
        return_latents: bool = False
    ) -> torch.Tensor:
        """
        Generate video with optimized pipeline.

        Args:
            prompt: Text prompt for generation
            num_frames: Number of frames to generate
            seed: Random seed (optional)
            return_latents: Whether to return latents instead of pixels

        Returns:
            Video tensor [batch, frames, channels, height, width]
        """
        if not self._warmup_complete:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

        # Prefetch noise if enabled
        noise_shape = (1, num_frames, 16, 60, 104)
        if self.noise_prefetcher is not None:
            self.noise_prefetcher.prefetch(noise_shape, dtype=self.dtype)
            noise = self.noise_prefetcher.get()
        else:
            noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)

        # Run inference
        self.profiler.reset()
        start_time = time.time()

        with torch.no_grad():
            with self.profiler.profile_region("inference"):
                video = self.pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    return_latents=return_latents,
                    profile=False
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time

        # Update metrics
        self._total_frames += num_frames
        self._total_time += elapsed

        # Log performance
        fps = num_frames / elapsed
        ms_per_frame = elapsed * 1000 / num_frames
        print(f"Generated {num_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS, {ms_per_frame:.1f} ms/frame)")

        return video

    def generate_streaming(
        self,
        prompt: str,
        num_frames: int = 120,
        callback: Optional[callable] = None
    ):
        """
        Generate video with streaming output (frame-by-frame).

        Args:
            prompt: Text prompt
            num_frames: Total frames to generate
            callback: Function called for each decoded frame batch

        Yields:
            Video frames as they're decoded
        """
        if not self._warmup_complete:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        num_blocks = num_frames // self.opt_config.num_frame_per_block

        for block_idx in range(num_blocks):
            # Generate block
            block_frames = self.opt_config.num_frame_per_block
            noise = torch.randn(
                [1, block_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )

            with torch.no_grad():
                block_video = self.pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    return_latents=False,
                    profile=False
                )

            # Yield or callback
            if callback is not None:
                callback(block_video, block_idx)

            yield block_video

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        avg_fps = self._total_frames / self._total_time if self._total_time > 0 else 0
        avg_ms = self._total_time * 1000 / self._total_frames if self._total_frames > 0 else 0

        metrics = {
            "total_frames": self._total_frames,
            "total_time_s": self._total_time,
            "average_fps": avg_fps,
            "average_ms_per_frame": avg_ms,
            "warmup_complete": self._warmup_complete,
        }

        if self.kv_cache is not None:
            metrics["kv_cache_memory_mb"] = self.kv_cache.memory_usage_mb()

        if self.device.type == "cuda":
            metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

        return metrics

    def get_profiling_report(self, format: str = "text") -> str:
        """Get profiling report."""
        return self.profiler.generate_report(format=format)

    def reset_metrics(self):
        """Reset performance metrics."""
        self._total_frames = 0
        self._total_time = 0.0
        self.profiler.reset()

    def cleanup(self):
        """Cleanup resources."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.kv_cache = None
        self.streaming_vae = None
        self.prefix_cache = None
        self.noise_prefetcher = None
        self.triple_buffer = None


def main():
    """CLI entry point for optimized inference."""
    parser = argparse.ArgumentParser(description="LongLive Optimized GPU Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean",
                        help="Generation prompt")
    parser.add_argument("--num-frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Optimization flags
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 quantization")
    parser.add_argument("--no-quantized-kv", action="store_true", help="Disable quantized KV cache")
    parser.add_argument("--local-attn-size", type=int, default=9, help="Local attention window")

    args = parser.parse_args()

    # Create optimization config
    opt_config = OptimizationConfig(
        use_torch_compile=not args.no_compile,
        use_fp8=not args.no_fp8,
        use_quantized_kv=not args.no_quantized_kv,
        local_attn_size=args.local_attn_size,
    )

    # Run inference
    pipeline = OptimizedInferencePipeline(args.config, opt_config)
    pipeline.setup()

    video = pipeline.generate(
        prompt=args.prompt,
        num_frames=args.num_frames,
        seed=args.seed
    )

    # Save video
    print(f"\nSaving video to {args.output}...")
    import imageio
    video_np = (video[0] * 255).to(torch.uint8).cpu().numpy()
    video_np = video_np.transpose(0, 2, 3, 1)  # [T, H, W, C]
    imageio.mimsave(args.output, video_np, fps=16)

    # Print metrics
    metrics = pipeline.get_metrics()
    print(f"\nPerformance Summary:")
    print(f"  Average FPS: {metrics['average_fps']:.2f}")
    print(f"  Average ms/frame: {metrics['average_ms_per_frame']:.1f}")
    if 'gpu_memory_allocated_gb' in metrics:
        print(f"  GPU Memory: {metrics['gpu_memory_allocated_gb']:.1f} GB")

    pipeline.cleanup()


if __name__ == "__main__":
    main()
