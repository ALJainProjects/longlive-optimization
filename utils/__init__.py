# LongLive Optimization Utilities
# SPDX-License-Identifier: Apache-2.0
"""
Utility modules for LongLive optimization.

This package provides:
- Profiling: Fine-grained latency measurement with CUDA events
- Quantization: FP8/INT8 support, torch.compile, CUDA graphs
- Advanced Optimizations: KV cache, streaming VAE, token merging
- GPU Workflow: Integrated optimized inference pipeline
"""

from utils.profiler import LongLiveProfiler, get_profiler, set_profiler, profile_region
from utils.quantization import (
    apply_fp8_quantization,
    apply_int8_quantization,
    apply_torch_compile,
    CUDAGraphWrapper,
    get_model_size_mb,
)
from utils.advanced_optimizations import (
    QuantizedKVCache,
    StreamingVAEDecoder,
    PrefixCache,
    TripleBuffer,
    AsyncNoisePrefetcher,
    merge_tokens,
    unmerge_tokens,
    apply_all_optimizations,
)

__all__ = [
    # Profiling
    "LongLiveProfiler",
    "get_profiler",
    "set_profiler",
    "profile_region",
    # Quantization
    "apply_fp8_quantization",
    "apply_int8_quantization",
    "apply_torch_compile",
    "CUDAGraphWrapper",
    "get_model_size_mb",
    # Advanced Optimizations
    "QuantizedKVCache",
    "StreamingVAEDecoder",
    "PrefixCache",
    "TripleBuffer",
    "AsyncNoisePrefetcher",
    "merge_tokens",
    "unmerge_tokens",
    "apply_all_optimizations",
]
