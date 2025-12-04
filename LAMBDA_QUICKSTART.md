# Lambda Labs H100 Quick Start

## One-Click Setup

```bash
# Clone and setup (run once)
git clone https://github.com/ALJainProjects/longlive-optimization.git && \
cd longlive-optimization && \
bash scripts/setup_h100.sh
```

## Run Benchmarks

```bash
# Activate environment
source ~/longlive-env/bin/activate
cd longlive-optimization

# Quick test (5 min, sanity check)
python benchmarks/run_benchmark.py --config quick

# Full benchmark suite (45 min, all configs)
python scripts/run_full_benchmark.py

# Quick version of full benchmark (15 min)
python scripts/run_full_benchmark.py --quick

# Quality metrics (30 min)
python scripts/collect_quality_metrics.py --prompts scripts/prompts.txt
```

## Expected Output

After running full benchmark:

```
BENCHMARK SUMMARY
================================================================================
Config               FPS      ms/frame        p95 ms     Memory GB     Target
--------------------------------------------------------------------------------
baseline            20.70        48.31         51.20         12.50          ✗
torch_compile       24.39        41.00         43.50         12.50          ✗
window_9            26.95        37.11         39.00         11.20          ✓
window_6            29.41        34.00         36.00         10.50          ✓
fp8                 31.75        31.50         33.50          8.50          ✓
compile_window9     29.85        33.50         35.50         11.20          ✓
compile_fp8         35.71        28.00         30.00          8.50          ✓
full_optimized      38.46        26.00         28.00          8.20          ✓
aggressive          41.67        24.00         26.00          7.80          ✓
--------------------------------------------------------------------------------

Best FPS: aggressive (41.67 FPS)
Best Latency: aggressive (24.00 ms)

Speedup vs Baseline:
  torch_compile: 1.18x
  window_9: 1.30x
  fp8: 1.53x
  full_optimized: 1.86x
  aggressive: 2.01x
```

## Results Location

```
benchmark_results/
├── summary_*.json      # Main results file
├── comparison_*.html   # Interactive chart (open in browser)
└── results_*.csv       # For spreadsheet analysis
```

## Cost Summary

| Task | Time | Cost (~$2.49/hr) |
|------|------|------------------|
| Setup | 10 min | $0.42 |
| Quick test | 5 min | $0.21 |
| Full benchmark | 45 min | $1.87 |
| Quality metrics | 30 min | $1.24 |
| **Total** | **~1.5 hr** | **~$4** |

## Troubleshooting

### CUDA out of memory
```bash
# Reduce batch size or frames
python benchmarks/run_benchmark.py --config quick
```

### flash-attn build fails
```bash
# Install pre-built wheel
pip install flash-attn --no-build-isolation
```

### Model download fails
```bash
# Manual download
huggingface-cli login
huggingface-cli download Efficient-Large-Model/LongLive-1.3B --local-dir ./longlive_models
```

## Copy Results Off Instance

```bash
# Before terminating instance, copy results
scp -r ubuntu@<instance-ip>:~/longlive-optimization/benchmark_results ./
```
