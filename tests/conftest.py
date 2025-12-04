# LongLive Test Fixtures
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for LongLive optimization tests."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Default dtype for testing."""
    return torch.float32


@pytest.fixture
def sample_tensor(device, dtype):
    """Create a sample tensor for testing."""
    return torch.randn(2, 16, 64, device=device, dtype=dtype)


@pytest.fixture
def sample_batch(device, dtype):
    """Create a sample batch tensor for testing."""
    return torch.randn(2, 8, 128, device=device, dtype=dtype)


@pytest.fixture
def simple_linear_model():
    """Create a simple linear model for quantization tests."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.linear2 = nn.Linear(128, 64)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.linear1(x))
            return self.linear2(x)

    return SimpleModel()


@pytest.fixture
def multi_layer_model():
    """Create a multi-layer model for testing."""
    class MultiLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64) for _ in range(4)
            ])
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.norm(x)

    return MultiLayerModel()


@pytest.fixture
def mock_benchmark_results() -> Dict[str, Any]:
    """Create mock benchmark results for visualization tests."""
    return {
        "config": {
            "model_name": "Wan2.1-T2V-1.3B",
            "num_output_frames": 120,
            "batch_size": 1,
            "local_attn_size": 12,
            "denoising_steps": [1000, 750, 500, 250],
            "gpu_name": "Test GPU",
            "use_torch_compile": False,
            "use_fp8": False,
        },
        "summary": {
            "fps": {"mean": 20.0, "std": 1.0, "min": 18.5, "max": 21.5},
            "ms_per_frame": {"mean": 50.0, "std": 2.5, "p50": 49.0, "p95": 54.0, "p99": 55.0},
            "memory_peak_gb": {"mean": 8.5, "max": 9.0},
            "steady_state_inter_frame_ms": {"mean": 48.0},
            "num_iterations": 10,
            "num_frames_per_iteration": 120,
        },
        "iterations": [
            {"iteration": i, "fps": 20.0 + np.random.randn(), "ms_per_frame": 50.0 + np.random.randn()}
            for i in range(10)
        ]
    }


@pytest.fixture
def mock_profiler_events():
    """Create mock profiler events for testing."""
    return {
        "inference": [
            {"start": 0.0, "end": 5.0},
            {"start": 5.0, "end": 10.0},
        ],
        "diffusion": [
            {"start": 0.1, "end": 4.0},
            {"start": 5.1, "end": 9.0},
        ],
        "block_0": [
            {"start": 0.1, "end": 0.5},
            {"start": 5.1, "end": 5.5},
        ],
        "block_1": [
            {"start": 0.5, "end": 0.9},
            {"start": 5.5, "end": 5.9},
        ],
        "block_2": [
            {"start": 0.9, "end": 1.3},
            {"start": 5.9, "end": 6.3},
        ],
    }


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Reset CUDA memory before and after each test."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
