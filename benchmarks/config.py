# LongLive Benchmark Configuration
# SPDX-License-Identifier: Apache-2.0
"""
Reproducible benchmark configurations for LongLive video generation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import yaml
import json
import torch


@dataclass
class BenchmarkConfig:
    """Reproducible benchmark configuration."""

    # Model configuration
    model_name: str = "Wan2.1-T2V-1.3B"
    local_attn_size: int = 12
    sink_size: int = 3
    num_transformer_blocks: int = 30
    frame_seq_length: int = 1560

    # Denoising configuration
    denoising_steps: List[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    num_frame_per_block: int = 3
    warp_denoising_step: bool = True

    # Generation configuration
    num_output_frames: int = 120
    batch_size: int = 1
    latent_height: int = 60
    latent_width: int = 104
    latent_channels: int = 16

    # Benchmark settings
    warmup_iterations: int = 2
    benchmark_iterations: int = 10
    seed: int = 42

    # Profiling settings
    profile_layers: bool = True
    profile_memory: bool = True
    sync_cuda: bool = True

    # Optimization flags
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"  # or "max-autotune"
    use_fp8: bool = False
    use_cuda_graph: bool = False

    # Hardware info (auto-populated)
    gpu_name: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    flash_attn_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'BenchmarkConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> 'BenchmarkConfig':
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def populate_hardware_info(self):
        """Auto-populate hardware information."""
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.cuda_version = torch.version.cuda or "N/A"
        else:
            self.gpu_name = "N/A (CPU)"
            self.cuda_version = "N/A"

        self.torch_version = torch.__version__

        try:
            import flash_attn
            self.flash_attn_version = flash_attn.__version__
        except ImportError:
            self.flash_attn_version = "N/A"

    def validate(self):
        """Validate configuration parameters."""
        assert self.num_output_frames % self.num_frame_per_block == 0, \
            f"num_output_frames ({self.num_output_frames}) must be divisible by num_frame_per_block ({self.num_frame_per_block})"
        assert self.batch_size >= 1, "batch_size must be >= 1"
        assert self.warmup_iterations >= 0, "warmup_iterations must be >= 0"
        assert self.benchmark_iterations >= 1, "benchmark_iterations must be >= 1"
        assert self.local_attn_size == -1 or self.local_attn_size > 0, \
            "local_attn_size must be -1 (global) or positive"


# Predefined benchmark configurations
BENCHMARK_CONFIGS = {
    "standard": BenchmarkConfig(
        num_output_frames=120,
        batch_size=1,
        benchmark_iterations=10,
    ),
    "quick": BenchmarkConfig(
        num_output_frames=24,
        batch_size=1,
        warmup_iterations=1,
        benchmark_iterations=3,
    ),
    "long_video": BenchmarkConfig(
        num_output_frames=480,
        batch_size=1,
        benchmark_iterations=5,
    ),
    "window_comparison": BenchmarkConfig(
        num_output_frames=120,
        batch_size=1,
        benchmark_iterations=5,
        # Will be modified for different window sizes
    ),
    "fp8_test": BenchmarkConfig(
        num_output_frames=120,
        batch_size=1,
        benchmark_iterations=5,
        use_fp8=True,
    ),
    "compiled": BenchmarkConfig(
        num_output_frames=120,
        batch_size=1,
        benchmark_iterations=5,
        use_torch_compile=True,
        compile_mode="reduce-overhead",
    ),
}


def create_optimization_sweep() -> List[BenchmarkConfig]:
    """Create a sweep of configurations for optimization comparison."""
    configs = []

    # Baseline
    baseline = BenchmarkConfig(num_output_frames=120, benchmark_iterations=5)
    baseline.populate_hardware_info()
    configs.append(("baseline", baseline))

    # torch.compile
    compiled = BenchmarkConfig(
        num_output_frames=120,
        benchmark_iterations=5,
        use_torch_compile=True,
        compile_mode="reduce-overhead"
    )
    compiled.populate_hardware_info()
    configs.append(("torch_compile", compiled))

    # Window size variations
    for window_size in [6, 9, 12]:
        cfg = BenchmarkConfig(
            num_output_frames=120,
            benchmark_iterations=5,
            local_attn_size=window_size,
        )
        cfg.populate_hardware_info()
        configs.append((f"window_{window_size}", cfg))

    # FP8
    fp8_cfg = BenchmarkConfig(
        num_output_frames=120,
        benchmark_iterations=5,
        use_fp8=True,
    )
    fp8_cfg.populate_hardware_info()
    configs.append(("fp8", fp8_cfg))

    return configs
