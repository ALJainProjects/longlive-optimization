# LongLive Quantization Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization utilities."""

import pytest
import torch
import torch.nn as nn

from utils.quantization import (
    apply_torch_compile,
    CUDAGraphWrapper,
    get_model_size_mb,
    print_optimization_summary,
)


class TestApplyTorchCompile:
    """Tests for torch.compile wrapper."""

    def test_compile_simple_model(self, simple_linear_model):
        """Test compilation of simple model."""
        model = simple_linear_model

        compiled = apply_torch_compile(model, mode="default")

        # The model should still be callable
        x = torch.randn(2, 64)
        output = compiled(x)
        assert output.shape == (2, 64)

    def test_compile_with_different_modes(self, simple_linear_model):
        """Test different compilation modes."""
        for mode in ["default", "reduce-overhead"]:
            model = type(simple_linear_model)()
            compiled = apply_torch_compile(model, mode=mode)
            x = torch.randn(2, 64)
            output = compiled(x)
            assert output.shape == (2, 64)

    def test_compile_with_fullgraph(self, simple_linear_model):
        """Test compilation with fullgraph option."""
        compiled = apply_torch_compile(
            simple_linear_model,
            mode="default",
            fullgraph=False,
            dynamic=True
        )

        x = torch.randn(2, 64)
        output = compiled(x)
        assert output.shape == (2, 64)


class TestCUDAGraphWrapper:
    """Tests for CUDA graph wrapper."""

    def test_wrapper_init(self, simple_linear_model):
        """Test wrapper initialization."""
        wrapper = CUDAGraphWrapper(simple_linear_model)
        assert wrapper.model is simple_linear_model
        assert wrapper.graph is None
        assert not wrapper._captured

    def test_wrapper_forward_uncaptured(self, simple_linear_model, device):
        """Test forward pass without capture."""
        model = simple_linear_model.to(device)
        wrapper = CUDAGraphWrapper(model)

        x = torch.randn(2, 64, device=device)
        output = wrapper(x)

        assert output.shape == (2, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wrapper_warmup_and_capture(self, simple_linear_model):
        """Test warmup and CUDA graph capture."""
        device = torch.device("cuda")
        model = simple_linear_model.to(device)
        wrapper = CUDAGraphWrapper(model)

        sample_input = torch.randn(2, 64, device=device)
        wrapper.warmup(sample_input, num_warmup=3)

        assert wrapper._captured
        assert wrapper.graph is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wrapper_forward_captured(self, simple_linear_model):
        """Test forward pass with captured graph."""
        device = torch.device("cuda")
        model = simple_linear_model.to(device)
        wrapper = CUDAGraphWrapper(model)

        sample_input = torch.randn(2, 64, device=device)
        wrapper.warmup(sample_input, num_warmup=3)

        # Forward with captured graph
        x = torch.randn(2, 64, device=device)
        output = wrapper(x)

        assert output.shape == (2, 64)

    def test_wrapper_warmup_cpu_fallback(self, simple_linear_model):
        """Test warmup on CPU (should not capture graph)."""
        model = simple_linear_model
        wrapper = CUDAGraphWrapper(model)

        sample_input = torch.randn(2, 64)
        wrapper.warmup(sample_input, num_warmup=3)

        # On CPU, graph capture should be skipped
        # Model should still work
        x = torch.randn(2, 64)
        output = wrapper(x)
        assert output.shape == (2, 64)


class TestModelSizeMB:
    """Tests for model size calculation."""

    def test_simple_model_size(self, simple_linear_model):
        """Test size calculation for simple model."""
        size_mb = get_model_size_mb(simple_linear_model)

        # Linear1: 64*128 + 128 = 8320 params
        # Linear2: 128*64 + 64 = 8256 params
        # Total: 16576 params * 4 bytes = ~0.063 MB
        assert size_mb > 0
        assert size_mb < 1.0  # Should be small

    def test_larger_model_size(self, multi_layer_model):
        """Test size calculation for larger model."""
        size_mb = get_model_size_mb(multi_layer_model)
        assert size_mb > 0

    def test_model_with_buffers(self):
        """Test size calculation includes buffers."""
        class ModelWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
                self.register_buffer('buf', torch.randn(100, 100))

            def forward(self, x):
                return self.linear(x)

        model = ModelWithBuffer()
        size_mb = get_model_size_mb(model)

        # Buffer alone: 100*100*4 bytes = 0.038 MB
        assert size_mb > 0.03


class TestPrintOptimizationSummary:
    """Tests for optimization summary printing."""

    def test_summary_prints(self, simple_linear_model, capsys):
        """Test that summary prints correctly."""
        original_size = get_model_size_mb(simple_linear_model)

        print_optimization_summary(
            simple_linear_model,
            original_size_mb=original_size,
            use_compile=True,
            use_fp8=False,
        )

        captured = capsys.readouterr()
        assert "OPTIMIZATION SUMMARY" in captured.out
        assert "torch.compile: Enabled" in captured.out
        assert "FP8 quantization: Disabled" in captured.out

    def test_summary_with_all_options(self, simple_linear_model, capsys):
        """Test summary with all optimizations enabled."""
        original_size = get_model_size_mb(simple_linear_model)

        print_optimization_summary(
            simple_linear_model,
            original_size_mb=original_size,
            use_compile=True,
            use_fp8=True,
            use_int8=False,
            use_cuda_graph=True,
        )

        captured = capsys.readouterr()
        assert "torch.compile: Enabled" in captured.out
        assert "FP8 quantization: Enabled" in captured.out
        assert "CUDA graph: Enabled" in captured.out
        assert "Estimated speedup" in captured.out


# Optional tests that require torchao
class TestFP8Quantization:
    """Tests for FP8 quantization (optional, requires torchao)."""

    @pytest.mark.skip(reason="Requires torchao installation")
    def test_fp8_quantization(self, simple_linear_model):
        """Test FP8 quantization application."""
        from utils.quantization import apply_fp8_quantization

        model = apply_fp8_quantization(simple_linear_model)

        x = torch.randn(2, 64)
        output = model(x)
        assert output.shape == (2, 64)


class TestINT8Quantization:
    """Tests for INT8 quantization (optional, requires torchao)."""

    @pytest.mark.skip(reason="Requires torchao installation")
    def test_int8_quantization(self, simple_linear_model):
        """Test INT8 quantization application."""
        from utils.quantization import apply_int8_quantization

        model = apply_int8_quantization(simple_linear_model)

        x = torch.randn(2, 64)
        output = model(x)
        assert output.shape == (2, 64)
