# LongLive Benchmark Scripts

Scripts for running benchmarks on Lambda Labs H100.

## Quick Start (Lambda Labs)

```bash
# 1. Clone repository
git clone https://github.com/ALJainProjects/longlive-optimization.git
cd longlive-optimization

# 2. Run setup (installs all dependencies + downloads models)
bash scripts/setup_h100.sh

# 3. Activate environment
source ~/longlive-env/bin/activate

# 4. Quick sanity test (5 minutes)
python benchmarks/run_benchmark.py --config quick

# 5. Full benchmark suite (30-60 minutes)
python scripts/run_full_benchmark.py

# 6. Quality metrics (optional, 30+ minutes)
python scripts/collect_quality_metrics.py
```

## Scripts

### `setup_h100.sh`
One-time setup script for Lambda Labs H100 instance:
- Creates Python virtual environment
- Installs PyTorch, flash-attn, torchao
- Downloads LongLive model weights
- Installs package in dev mode

### `run_full_benchmark.py`
Comprehensive benchmark across all optimization configurations:
- Baseline (no optimizations)
- torch.compile only
- Window size variations (12, 9, 6)
- FP8 quantization
- Combined optimizations
- Full optimized (recommended)
- Aggressive optimization

Output:
- JSON results per configuration
- Summary CSV
- HTML comparison chart

Options:
```bash
python scripts/run_full_benchmark.py --help
  --output-dir    Output directory (default: benchmark_results)
  --config        Base config file
  --quick         Quick mode (fewer frames/iterations)
```

### `collect_quality_metrics.py`
Quality evaluation for generated videos:
- CLIP Score (text-video alignment)
- Temporal Consistency (frame coherence)

Options:
```bash
python scripts/collect_quality_metrics.py --help
  --config        Config file
  --prompts       File with prompts (one per line)
  --output-dir    Output directory
  --configs       Which configs to test (baseline, optimized, aggressive)
```

## Expected Results

On H100 GPU:

| Configuration | FPS | ms/frame | Target (40ms) |
|---------------|-----|----------|---------------|
| Baseline | ~21 | ~48 | ✗ |
| + torch.compile | ~24 | ~41 | Near |
| + window=9 | ~27 | ~37 | ✓ |
| + FP8 | ~32 | ~31 | ✓ |
| Full Optimized | ~35-40 | ~25-28 | ✓ |

## Output Files

```
benchmark_results/
├── baseline_20240101_120000.json      # Individual config results
├── torch_compile_20240101_120000.json
├── ...
├── summary_20240101_120000.json       # Summary with all configs
├── comparison_20240101_120000.html    # Interactive chart
└── results_20240101_120000.csv        # CSV for analysis

quality_results/
└── quality_metrics.json               # CLIP + temporal scores
```

## Cost Estimate

Lambda Labs H100 (~$2.49/hr):
- Setup: ~10 minutes
- Quick benchmark: ~5 minutes
- Full benchmark: ~45 minutes
- Quality metrics: ~30 minutes

**Total: ~1.5 hours = ~$4**
