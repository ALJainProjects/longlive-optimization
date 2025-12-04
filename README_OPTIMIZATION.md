# LongLive Latency Optimization

[![CI](https://github.com/arnavjain/longlive-optimization/workflows/CI/badge.svg)](https://github.com/arnavjain/longlive-optimization/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains optimizations and analysis for [LongLive](https://github.com/NVlabs/LongLive), a real-time interactive video generation model. Target: **<40ms latency** for interactive use.

## Quick Start

```bash
# Install as package
pip install -e ".[all]"

# Or install core dependencies only
pip install -r requirements.txt
pip install flash-attn torchao

# Download model weights
huggingface-cli download Efficient-Large-Model/LongLive-1.3B

# Run optimized GPU inference
python -m utils.gpu_workflow \
    --config configs/longlive_optimized.yaml \
    --prompt "A cat walking on the beach" \
    --output video.mp4

# Run benchmarks
python benchmarks/run_benchmark.py --config standard
```

## Installation

```bash
# Development installation (recommended)
pip install -e ".[dev]"

# Install with demo dependencies
pip install -e ".[demo]"

# Install with all optional dependencies
pip install -e ".[all]"
```

## Project Structure

```
longlive-optimization/
├── benchmarks/                  # Benchmark infrastructure
│   ├── __init__.py             # Package exports
│   ├── config.py               # BenchmarkConfig dataclass
│   ├── runner.py               # BenchmarkRunner
│   ├── run_benchmark.py        # CLI entry point
│   └── visualization.py        # Time budget reports (text/JSON/HTML)
├── configs/
│   ├── longlive_inference.yaml # Original config
│   └── longlive_optimized.yaml # Optimized config
├── demo/                        # Interactive WebRTC demo
│   ├── __init__.py
│   ├── server.py               # FastAPI + WebRTC server
│   └── pipeline_wrapper.py     # Async pipeline wrapper
├── docs/
│   ├── LATENCY_ANALYSIS.md     # Detailed latency breakdown
│   ├── THOUGHT_EXERCISE.md     # Architectural analysis
│   └── ADDITIONAL_OPTIMIZATIONS.md  # Advanced optimization ideas
├── tests/                       # Comprehensive test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── test_profiler.py        # Profiler tests
│   ├── test_quantization.py    # Quantization tests
│   ├── test_advanced_optimizations.py
│   ├── test_benchmark_config.py
│   ├── test_visualization.py
│   ├── test_runner.py
│   └── test_demo.py
├── utils/
│   ├── __init__.py             # Package exports
│   ├── profiler.py             # LongLiveProfiler
│   ├── quantization.py         # FP8/INT8/CUDA Graph support
│   ├── advanced_optimizations.py  # KV cache, streaming VAE, etc.
│   └── gpu_workflow.py         # Integrated GPU pipeline
├── .github/workflows/           # CI/CD
│   ├── ci.yml                  # Tests, lint, build
│   └── release.yml             # PyPI publishing
├── pyproject.toml              # Package configuration
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
├── requirements-demo.txt       # Demo dependencies
└── README_OPTIMIZATION.md      # This file
```

## Optimizations Applied

### Core Optimizations (`utils/quantization.py`)

| Optimization | Expected Gain | Implementation |
|--------------|---------------|----------------|
| torch.compile | 10-20% | `apply_torch_compile()` |
| FP8 quantization | 15-20% | `apply_fp8_quantization()` |
| INT8 quantization | 10-15% | `apply_int8_quantization()` |
| CUDA Graph | 15-25% | `CUDAGraphWrapper` |
| Attention window 12→9 | 8-12% | Config change |

### Advanced Optimizations (`utils/advanced_optimizations.py`)

| Optimization | Benefit | Class/Function |
|--------------|---------|----------------|
| Quantized KV Cache | 2x memory reduction | `QuantizedKVCache` |
| Streaming VAE | Overlap decode/generate | `StreamingVAEDecoder` |
| Token Merging | Reduce sequence length | `merge_tokens()` |
| Prefix Caching | Skip redundant compute | `PrefixCache` |
| Triple Buffering | Pipeline stages | `TripleBuffer` |
| Async Noise Prefetch | Hide latency | `AsyncNoisePrefetcher` |

### Integrated GPU Workflow (`utils/gpu_workflow.py`)

All optimizations integrated into a single pipeline:

```python
from utils.gpu_workflow import OptimizedInferencePipeline, OptimizationConfig

config = OptimizationConfig(
    use_torch_compile=True,
    use_fp8=True,
    use_quantized_kv=True,
    local_attn_size=9,
)

pipeline = OptimizedInferencePipeline("configs/longlive_optimized.yaml", config)
pipeline.setup()
video = pipeline.generate("A beautiful sunset", num_frames=120)
```

## Benchmark Results (Expected)

| Configuration | FPS | ms/frame | Target Met? |
|---------------|-----|----------|-------------|
| Baseline | 20.7 | 48.3 | No |
| + torch.compile | 24.4 | 41.0 | Near |
| + window=9 | 27.0 | 37.1 | Yes |
| + FP8 | 31.7 | 31.5 | Yes |
| + All Advanced | 35-40 | 25-28 | Yes |

## Usage

### GPU Workflow (Recommended)

```bash
# Full optimized pipeline
python -m utils.gpu_workflow \
    --config configs/longlive_optimized.yaml \
    --prompt "A beautiful sunset over the ocean" \
    --num-frames 120 \
    --output video.mp4

# Customize optimizations
python -m utils.gpu_workflow \
    --config configs/longlive_optimized.yaml \
    --prompt "Your prompt" \
    --no-compile \
    --no-fp8 \
    --local-attn-size 12
```

### Run Benchmarks

```bash
# Standard benchmark
python benchmarks/run_benchmark.py --config standard

# Quick test
python benchmarks/run_benchmark.py --config quick

# Optimization comparison
python benchmarks/run_benchmark.py --comparison

# Custom settings
python benchmarks/run_benchmark.py --frames 60 --window-size 9 --compile --fp8
```

### Interactive Demo

```bash
# Start demo server
python demo/server.py --config configs/longlive_optimized.yaml --port 8000

# Or use the CLI entry point
longlive-demo --config configs/longlive_optimized.yaml
```

### Profiling

```python
from utils.profiler import LongLiveProfiler

profiler = LongLiveProfiler(enabled=True)

with profiler.profile_region("my_operation"):
    # Your code here
    pass

print(profiler.generate_report())
profiler.export_json("results.json")
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=utils --cov=benchmarks --cov=demo --cov-report=html

# Run specific test file
pytest tests/test_profiler.py -v

# Skip CUDA tests (when no GPU)
pytest tests/ -m "not cuda" -v
```

## Key Findings

### Latency Breakdown (Baseline 48ms/frame)
- Diffusion loop: 81% (~39ms)
- VAE decode: 10% (~5ms)
- Initialization: 6% (~3ms)
- Other: 3% (~1.5ms)

### Bottlenecks Identified
1. **Flash Attention** - 50% of diffusion time
2. **KV Cache Operations** - Memory bandwidth limited
3. **Sequential Frame Dependency** - Fundamental to autoregressive design

### Recommendations
- **Sweet spot:** window=9, FP8, torch.compile → 30-35 FPS
- **Maximum speed:** window=6, 2-step diffusion → 45-50 FPS (quality tradeoff)

## Files Created

| File | Description |
|------|-------------|
| `utils/profiler.py` | Fine-grained CUDA event profiling |
| `utils/quantization.py` | FP8/INT8/CUDA Graph support |
| `utils/advanced_optimizations.py` | KV cache, streaming VAE, token merging |
| `utils/gpu_workflow.py` | Integrated optimized pipeline |
| `benchmarks/*` | Complete benchmark infrastructure |
| `demo/*` | WebRTC streaming demo |
| `tests/*` | Comprehensive test suite |
| `docs/*` | Analysis documentation |
| `pyproject.toml` | Package configuration |
| `.github/workflows/*` | CI/CD pipelines |

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA
- H100 GPU (recommended for FP8)
- 40GB+ VRAM
- flash-attn
- torchao (for quantization)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
black .
isort .
flake8 utils benchmarks demo tests

# Run type checking
mypy utils benchmarks demo

# Run tests with coverage
pytest tests/ --cov --cov-report=html
```

## CI/CD

This project uses GitHub Actions for:
- **CI** (`.github/workflows/ci.yml`): Lint, test (Python 3.9/3.10/3.11), type check, build
- **Release** (`.github/workflows/release.yml`): Publish to PyPI on tag push

## License

Apache-2.0 (same as original LongLive)

## References

- [LongLive Paper](https://arxiv.org/abs/2509.22622)
- [LongLive Repository](https://github.com/NVlabs/LongLive)
- [Hugging Face Model](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request
