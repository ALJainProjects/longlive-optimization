# LongLive Quantization Utilities
# SPDX-License-Identifier: Apache-2.0
"""
Quantization support for LongLive video generation.

Supports:
- FP8 dynamic quantization (recommended for H100)
- INT8 static quantization (for broader GPU support)
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


def apply_fp8_quantization(
    model: nn.Module,
    filter_fn: Optional[Callable[[nn.Module, str], bool]] = None
) -> nn.Module:
    """
    Apply FP8 dynamic quantization to model.

    This uses TorchAO's float8 quantization which provides:
    - ~20% speedup on H100 GPUs
    - Marginal quality loss

    Args:
        model: The model to quantize
        filter_fn: Optional filter function (module, fqn) -> bool

    Returns:
        Quantized model
    """
    try:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
    except ImportError:
        print("Warning: torchao not installed. Install with: pip install torchao")
        return model

    if filter_fn is None:
        # Default: quantize all Linear layers except attention QKV
        def filter_fn(mod, fqn):
            if not isinstance(mod, nn.Linear):
                return False
            # Skip very small layers
            if mod.weight.numel() < 1024:
                return False
            return True

    try:
        quantize_(
            model,
            float8_dynamic_activation_float8_weight(),
            filter_fn=filter_fn
        )
        print("FP8 quantization applied successfully")
    except Exception as e:
        print(f"Warning: FP8 quantization failed: {e}")

    return model


def apply_int8_quantization(
    model: nn.Module,
    filter_fn: Optional[Callable[[nn.Module, str], bool]] = None
) -> nn.Module:
    """
    Apply INT8 dynamic quantization to model.

    This provides:
    - 1.9x model compression
    - Speedup on GPUs with INT8 tensor cores
    - Slightly more quality loss than FP8

    Args:
        model: The model to quantize
        filter_fn: Optional filter function (module, fqn) -> bool

    Returns:
        Quantized model
    """
    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
    except ImportError:
        print("Warning: torchao not installed. Install with: pip install torchao")
        return model

    if filter_fn is None:
        def filter_fn(mod, fqn):
            return isinstance(mod, nn.Linear)

    try:
        quantize_(
            model,
            int8_dynamic_activation_int8_weight(),
            filter_fn=filter_fn
        )
        print("INT8 quantization applied successfully")
    except Exception as e:
        print(f"Warning: INT8 quantization failed: {e}")

    return model


def apply_torch_compile(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True
) -> nn.Module:
    """
    Apply torch.compile to model for kernel fusion and optimization.

    Args:
        model: The model to compile
        mode: Compilation mode
            - "reduce-overhead": Fastest startup, good for inference
            - "max-autotune": Slowest startup, best performance
            - "default": Balanced
        fullgraph: Whether to require full graph capture
        dynamic: Whether to support dynamic shapes

    Returns:
        Compiled model
    """
    try:
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic
        )
        print(f"torch.compile applied with mode={mode}")
        return compiled_model
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
        return model


class CUDAGraphWrapper(nn.Module):
    """
    Wrapper for CUDA graph capture of a model.

    CUDA graphs capture a sequence of CUDA operations and replay them
    with minimal CPU overhead, providing 15-25% speedup.

    Usage:
        wrapper = CUDAGraphWrapper(model)
        wrapper.warmup(sample_input)
        output = wrapper(input)  # Uses CUDA graph
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.graph = None
        self.static_input = None
        self.static_output = None
        self._captured = False

    def warmup(self, sample_input: torch.Tensor, num_warmup: int = 3):
        """
        Warmup and capture CUDA graph.

        Args:
            sample_input: A sample input tensor
            num_warmup: Number of warmup iterations
        """
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, skipping graph capture")
            return

        device = sample_input.device
        dtype = sample_input.dtype

        # Allocate static tensors
        self.static_input = sample_input.clone()

        # Warmup
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            for _ in range(num_warmup):
                self.static_output = self.model(self.static_input)

        torch.cuda.current_stream().wait_stream(stream)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

        self._captured = True
        print("CUDA graph captured successfully")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using CUDA graph if captured."""
        if not self._captured:
            return self.model(x)

        # Copy input to static buffer
        self.static_input.copy_(x)

        # Replay graph
        self.graph.replay()

        # Return output (clone to avoid aliasing issues)
        return self.static_output.clone()


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def print_optimization_summary(
    model: nn.Module,
    original_size_mb: float,
    use_compile: bool = False,
    use_fp8: bool = False,
    use_int8: bool = False,
    use_cuda_graph: bool = False
):
    """Print summary of applied optimizations."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)

    current_size = get_model_size_mb(model)
    compression = original_size_mb / current_size if current_size > 0 else 1.0

    print(f"Model size: {current_size:.1f} MB (was {original_size_mb:.1f} MB, {compression:.2f}x compression)")
    print(f"torch.compile: {'Enabled' if use_compile else 'Disabled'}")
    print(f"FP8 quantization: {'Enabled' if use_fp8 else 'Disabled'}")
    print(f"INT8 quantization: {'Enabled' if use_int8 else 'Disabled'}")
    print(f"CUDA graph: {'Enabled' if use_cuda_graph else 'Disabled'}")

    # Estimated speedup
    speedup = 1.0
    if use_compile:
        speedup *= 1.15  # ~15% from compile
    if use_fp8:
        speedup *= 1.20  # ~20% from FP8
    if use_cuda_graph:
        speedup *= 1.20  # ~20% from CUDA graph

    print(f"Estimated speedup: {speedup:.2f}x")
    print("=" * 60)
