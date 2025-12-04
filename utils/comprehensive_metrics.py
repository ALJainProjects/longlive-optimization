# LongLive Comprehensive Metrics Module
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive metrics collection for LongLive benchmarking.

Includes:
- Component-level timing (attention, KV cache, FFN)
- Memory bandwidth estimation
- Cross-batch latency
- Prompt-switch latency
- Quality metrics (FVD, CLIP, temporal consistency)
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np


@dataclass
class ComponentTiming:
    """Timing data for a single component."""
    name: str
    total_ms: float = 0.0
    call_count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count > 0 else 0.0

    def record(self, duration_ms: float):
        self.total_ms += duration_ms
        self.call_count += 1
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms if self.min_ms != float('inf') else 0,
            "max_ms": self.max_ms,
            "call_count": self.call_count,
        }


class ModelInstrumentor:
    """
    Instrument model layers for detailed timing breakdown.

    Usage:
        instrumentor = ModelInstrumentor(model)
        instrumentor.start()
        # ... run inference ...
        instrumentor.stop()
        timing = instrumentor.get_timing()
    """

    def __init__(self, model: nn.Module, sync_cuda: bool = True):
        self.model = model
        self.sync_cuda = sync_cuda
        self.timings: Dict[str, ComponentTiming] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.active = False

        # Pre-create timing buckets
        self._init_timing_categories()

    def _init_timing_categories(self):
        """Initialize timing categories."""
        categories = [
            "self_attention_qkv",
            "self_attention_rope",
            "self_attention_flash",
            "self_attention_output",
            "cross_attention",
            "kv_cache_read",
            "kv_cache_write",
            "kv_cache_roll",
            "ffn",
            "layer_norm",
            "embeddings",
        ]
        for cat in categories:
            self.timings[cat] = ComponentTiming(name=cat)

    def start(self):
        """Start instrumentation."""
        self.active = True
        self._register_hooks()

    def stop(self):
        """Stop instrumentation and remove hooks."""
        self.active = False
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset(self):
        """Reset all timing data."""
        for timing in self.timings.values():
            timing.total_ms = 0.0
            timing.call_count = 0
            timing.min_ms = float('inf')
            timing.max_ms = 0.0

    def _register_hooks(self):
        """Register forward hooks on relevant layers."""
        # This is a simplified version - full implementation would
        # register hooks on specific attention/FFN sublayers
        pass

    @contextmanager
    def time_region(self, name: str):
        """Context manager to time a specific region."""
        if not self.active:
            yield
            return

        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        yield

        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        duration_ms = (time.perf_counter() - start) * 1000

        if name not in self.timings:
            self.timings[name] = ComponentTiming(name=name)
        self.timings[name].record(duration_ms)

    def get_timing(self) -> Dict[str, Dict[str, float]]:
        """Get all timing data."""
        return {name: timing.to_dict() for name, timing in self.timings.items()}


@dataclass
class MemoryMetrics:
    """Memory usage and bandwidth metrics."""
    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    kv_cache_size_gb: float = 0.0
    activation_memory_gb: float = 0.0

    # Bandwidth estimates
    estimated_memory_reads_gb: float = 0.0
    estimated_memory_writes_gb: float = 0.0
    estimated_bandwidth_util: float = 0.0  # Percentage of peak

    def to_dict(self) -> Dict[str, float]:
        return {
            "peak_allocated_gb": self.peak_allocated_gb,
            "peak_reserved_gb": self.peak_reserved_gb,
            "kv_cache_size_gb": self.kv_cache_size_gb,
            "activation_memory_gb": self.activation_memory_gb,
            "estimated_memory_reads_gb": self.estimated_memory_reads_gb,
            "estimated_memory_writes_gb": self.estimated_memory_writes_gb,
            "estimated_bandwidth_util_pct": self.estimated_bandwidth_util,
        }


def estimate_memory_bandwidth(
    num_blocks: int,
    num_timesteps: int,
    num_frames: int,
    frame_seq_length: int,
    local_attn_size: int,
    num_transformer_blocks: int = 30,
    hidden_dim: int = 1536,
    num_heads: int = 12,
    head_dim: int = 128,
    dtype_bytes: int = 2,  # bfloat16
    duration_s: float = 1.0,
    peak_bandwidth_tb_s: float = 3.35,  # H100 PCIe
) -> MemoryMetrics:
    """
    Estimate memory bandwidth utilization.

    Based on LongLive architecture:
    - Per transformer block: QKV projection, attention, FFN
    - KV cache reads/writes
    """
    metrics = MemoryMetrics()

    # KV cache size
    kv_cache_tokens = local_attn_size * frame_seq_length
    kv_cache_per_block = kv_cache_tokens * num_heads * head_dim * dtype_bytes * 2  # K and V
    total_kv_cache_bytes = kv_cache_per_block * num_transformer_blocks
    metrics.kv_cache_size_gb = total_kv_cache_bytes / (1024**3)

    # Per-step memory access estimate
    tokens_per_step = num_frames * frame_seq_length

    # QKV projection: read hidden, write Q, K, V
    qkv_read = tokens_per_step * hidden_dim * dtype_bytes
    qkv_write = tokens_per_step * 3 * num_heads * head_dim * dtype_bytes

    # Attention: read K, V cache, write output
    attn_read = kv_cache_tokens * num_heads * head_dim * dtype_bytes * 2  # K and V
    attn_write = tokens_per_step * num_heads * head_dim * dtype_bytes

    # FFN: read hidden, write hidden (simplified)
    ffn_read_write = tokens_per_step * hidden_dim * dtype_bytes * 4  # Up + down proj

    # Total per transformer block per timestep
    per_block_bytes = qkv_read + qkv_write + attn_read + attn_write + ffn_read_write

    # Total for all blocks, timesteps, and video blocks
    total_bytes = per_block_bytes * num_transformer_blocks * num_timesteps * num_blocks

    metrics.estimated_memory_reads_gb = total_bytes / 2 / (1024**3)  # Rough split
    metrics.estimated_memory_writes_gb = total_bytes / 2 / (1024**3)

    # Bandwidth utilization
    total_gb = (metrics.estimated_memory_reads_gb + metrics.estimated_memory_writes_gb)
    achieved_bandwidth_tb_s = total_gb / 1000 / duration_s if duration_s > 0 else 0
    metrics.estimated_bandwidth_util = (achieved_bandwidth_tb_s / peak_bandwidth_tb_s) * 100

    return metrics


@dataclass
class CrossBatchMetrics:
    """Metrics for cross-batch latency analysis."""
    first_batch_first_block_ms: float = 0.0
    subsequent_batch_first_block_ms: float = 0.0
    cold_start_overhead_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "first_batch_first_block_ms": self.first_batch_first_block_ms,
            "subsequent_batch_first_block_ms": self.subsequent_batch_first_block_ms,
            "cold_start_overhead_ms": self.cold_start_overhead_ms,
        }


@dataclass
class PromptSwitchMetrics:
    """Metrics for prompt switching latency."""
    recache_time_ms: float = 0.0
    crossattn_reset_ms: float = 0.0
    first_new_frame_ms: float = 0.0
    total_switch_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "recache_time_ms": self.recache_time_ms,
            "crossattn_reset_ms": self.crossattn_reset_ms,
            "first_new_frame_ms": self.first_new_frame_ms,
            "total_switch_latency_ms": self.total_switch_latency_ms,
        }


# ============================================================================
# Quality Metrics
# ============================================================================

@dataclass
class QualityMetrics:
    """Quality metrics for generated videos."""
    fvd_score: Optional[float] = None
    clip_score: Optional[float] = None
    temporal_consistency: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "fvd_score": self.fvd_score,
            "clip_score": self.clip_score,
            "temporal_consistency": self.temporal_consistency,
            "psnr": self.psnr,
            "ssim": self.ssim,
        }


def compute_clip_score(
    frames: torch.Tensor,  # [N, 3, H, W]
    prompt: str,
    clip_model: Optional[Any] = None,
    clip_processor: Optional[Any] = None,
) -> float:
    """
    Compute CLIP score between frames and prompt.

    Returns average cosine similarity across frames.
    """
    try:
        if clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        device = frames.device
        clip_model = clip_model.to(device)

        # Process text
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Get text features
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Process images and get scores
        scores = []
        for frame in frames:
            # Convert to PIL for CLIP processor
            frame_np = (frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            from PIL import Image
            frame_pil = Image.fromarray(frame_np)

            image_inputs = clip_processor(images=frame_pil, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()
            scores.append(similarity)

        return float(np.mean(scores))

    except ImportError:
        print("Warning: transformers not available for CLIP score")
        return 0.0
    except Exception as e:
        print(f"Warning: CLIP score computation failed: {e}")
        return 0.0


def compute_temporal_consistency(
    frames: torch.Tensor,  # [N, 3, H, W]
) -> float:
    """
    Compute temporal consistency as average frame-to-frame similarity.

    Uses structural similarity between consecutive frames.
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        scores = []
        frames_np = frames.cpu().numpy()

        for i in range(len(frames_np) - 1):
            frame1 = frames_np[i].transpose(1, 2, 0)
            frame2 = frames_np[i + 1].transpose(1, 2, 0)

            # Compute SSIM
            score = ssim(frame1, frame2, channel_axis=2, data_range=1.0)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    except ImportError:
        print("Warning: skimage not available for temporal consistency")
        return 0.0
    except Exception as e:
        print(f"Warning: Temporal consistency computation failed: {e}")
        return 0.0


def compute_psnr(
    generated: torch.Tensor,  # [N, 3, H, W]
    reference: torch.Tensor,   # [N, 3, H, W]
) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = torch.mean((generated - reference) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return float(psnr)


def compute_ssim_score(
    generated: torch.Tensor,  # [N, 3, H, W]
    reference: torch.Tensor,   # [N, 3, H, W]
) -> float:
    """Compute Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity as ssim

        gen_np = generated.cpu().numpy()
        ref_np = reference.cpu().numpy()

        scores = []
        for i in range(len(gen_np)):
            gen_frame = gen_np[i].transpose(1, 2, 0)
            ref_frame = ref_np[i].transpose(1, 2, 0)
            score = ssim(gen_frame, ref_frame, channel_axis=2, data_range=1.0)
            scores.append(score)

        return float(np.mean(scores))

    except ImportError:
        print("Warning: skimage not available for SSIM")
        return 0.0


def compute_fvd(
    generated_videos: List[torch.Tensor],  # List of [T, 3, H, W]
    reference_videos: List[torch.Tensor],   # List of [T, 3, H, W]
    i3d_model: Optional[Any] = None,
) -> float:
    """
    Compute Frechet Video Distance.

    Note: FVD requires I3D features and a reference distribution.
    This is a simplified implementation that returns 0 if dependencies missing.
    """
    try:
        # Full FVD implementation requires:
        # 1. Pre-trained I3D model
        # 2. Feature extraction from both sets
        # 3. Frechet distance computation

        # For now, return placeholder
        print("Warning: Full FVD computation requires I3D model - returning placeholder")
        return 0.0

    except Exception as e:
        print(f"Warning: FVD computation failed: {e}")
        return 0.0


# ============================================================================
# Comprehensive Metrics Collection
# ============================================================================

@dataclass
class ComprehensiveMetrics:
    """All metrics collected during a benchmark run."""

    # Latency metrics
    batch_fps: float = 0.0
    batch_ms_per_frame: float = 0.0
    steady_state_fps: float = 0.0
    steady_state_ms: float = 0.0
    time_to_first_frame_ms: float = 0.0

    # Component breakdown
    init_time_ms: float = 0.0
    diffusion_time_ms: float = 0.0
    vae_time_ms: float = 0.0

    # Detailed component timing
    component_timing: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Memory metrics
    memory_metrics: Optional[MemoryMetrics] = None

    # Cross-batch metrics
    cross_batch_metrics: Optional[CrossBatchMetrics] = None

    # Prompt switch metrics (if applicable)
    prompt_switch_metrics: Optional[PromptSwitchMetrics] = None

    # Quality metrics
    quality_metrics: Optional[QualityMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "latency": {
                "batch_fps": self.batch_fps,
                "batch_ms_per_frame": self.batch_ms_per_frame,
                "steady_state_fps": self.steady_state_fps,
                "steady_state_ms": self.steady_state_ms,
                "time_to_first_frame_ms": self.time_to_first_frame_ms,
            },
            "component_breakdown": {
                "init_time_ms": self.init_time_ms,
                "diffusion_time_ms": self.diffusion_time_ms,
                "vae_time_ms": self.vae_time_ms,
            },
            "detailed_timing": self.component_timing,
        }

        if self.memory_metrics:
            result["memory"] = self.memory_metrics.to_dict()

        if self.cross_batch_metrics:
            result["cross_batch"] = self.cross_batch_metrics.to_dict()

        if self.prompt_switch_metrics:
            result["prompt_switch"] = self.prompt_switch_metrics.to_dict()

        if self.quality_metrics:
            result["quality"] = self.quality_metrics.to_dict()

        return result
