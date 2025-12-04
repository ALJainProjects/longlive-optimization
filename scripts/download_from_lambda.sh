#!/bin/bash
# Download results from running Lambda Labs instance
#
# Usage: bash scripts/download_from_lambda.sh <instance_ip>
#
# Example: bash scripts/download_from_lambda.sh 34.123.45.67

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/download_from_lambda.sh <instance_ip>"
    exit 1
fi

INSTANCE_IP=$1
LOCAL_DIR="./lambda_results_$(date +%Y%m%d_%H%M%S)"

echo "Downloading results from $INSTANCE_IP..."
mkdir -p "$LOCAL_DIR"

# Download benchmark results
echo "Downloading benchmark_results..."
scp -r ubuntu@$INSTANCE_IP:~/longlive-optimization/benchmark_results "$LOCAL_DIR/" 2>/dev/null || echo "  (no benchmark_results found)"

# Download comparison videos
echo "Downloading comparison_videos..."
scp -r ubuntu@$INSTANCE_IP:~/longlive-optimization/comparison_videos "$LOCAL_DIR/" 2>/dev/null || echo "  (no comparison_videos found)"

# Download quality results
echo "Downloading quality_results..."
scp -r ubuntu@$INSTANCE_IP:~/longlive-optimization/quality_results "$LOCAL_DIR/" 2>/dev/null || echo "  (no quality_results found)"

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo ""
echo "Results saved to: $LOCAL_DIR"
echo ""
echo "Contents:"
ls -la "$LOCAL_DIR"
echo ""
echo "View comparison videos:"
echo "  open $LOCAL_DIR/comparison_videos/comparison.html"
echo ""
echo "View benchmark summary:"
echo "  cat $LOCAL_DIR/benchmark_results/summary_*.json"
