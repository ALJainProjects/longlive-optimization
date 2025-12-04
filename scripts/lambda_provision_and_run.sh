#!/bin/bash
# Lambda Labs H100 Provisioning and Benchmark Script
#
# This script:
# 1. Provisions an H100 instance via Lambda Labs API
# 2. Waits for it to be ready
# 3. Runs setup and benchmarks
# 4. Downloads results locally
# 5. Terminates the instance
#
# Usage: bash scripts/lambda_provision_and_run.sh
#
# Requires: LAMBDA_API_KEY environment variable (or source ~/.ray-cluster-credentials)

set -e

# Load credentials
if [ -f ~/.ray-cluster-credentials ]; then
    source ~/.ray-cluster-credentials
fi

if [ -z "$LAMBDA_API_KEY" ]; then
    echo "ERROR: LAMBDA_API_KEY not set"
    echo "Set it with: export LAMBDA_API_KEY=your_key"
    echo "Or source ~/.ray-cluster-credentials"
    exit 1
fi

# Configuration
INSTANCE_TYPE="gpu_1x_h100_pcie"  # Single H100
REGION="us-west-3"  # H100 PCIe available here
SSH_KEY_NAME="lambda-gh200"  # Your SSH key name in Lambda Labs
LOCAL_RESULTS_DIR="./lambda_results"
REPO_URL="https://github.com/ALJainProjects/longlive-optimization.git"

echo "=============================================="
echo "Lambda Labs H100 Provisioning Script"
echo "=============================================="
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo ""

# Check for available instances
echo "[1/7] Checking instance availability..."
AVAILABILITY=$(curl -s -u "$LAMBDA_API_KEY:" \
    "https://cloud.lambdalabs.com/api/v1/instance-types" | \
    python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data.get('data', {}).get('$INSTANCE_TYPE', {}), indent=2))")

echo "$AVAILABILITY"

# Launch instance
echo ""
echo "[2/7] Launching H100 instance..."
LAUNCH_RESPONSE=$(curl -s -u "$LAMBDA_API_KEY:" \
    -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/launch" \
    -H "Content-Type: application/json" \
    -d "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        \"quantity\": 1
    }")

echo "$LAUNCH_RESPONSE"

# Extract instance ID
INSTANCE_ID=$(echo "$LAUNCH_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); ids=data.get('data', {}).get('instance_ids', []); print(ids[0] if ids else '')")

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Failed to launch instance"
    echo "$LAUNCH_RESPONSE"
    exit 1
fi

echo "Instance ID: $INSTANCE_ID"

# Wait for instance to be ready
echo ""
echo "[3/7] Waiting for instance to be ready..."
MAX_WAIT=300  # 5 minutes
WAITED=0
INSTANCE_IP=""

while [ $WAITED -lt $MAX_WAIT ]; do
    INSTANCE_INFO=$(curl -s -u "$LAMBDA_API_KEY:" \
        "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID")

    STATUS=$(echo "$INSTANCE_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('data', {}).get('status', 'unknown'))")
    INSTANCE_IP=$(echo "$INSTANCE_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('data', {}).get('ip', ''))")

    echo "  Status: $STATUS (waited ${WAITED}s)"

    if [ "$STATUS" == "active" ] && [ -n "$INSTANCE_IP" ]; then
        echo "  Instance ready! IP: $INSTANCE_IP"
        break
    fi

    sleep 10
    WAITED=$((WAITED + 10))
done

if [ -z "$INSTANCE_IP" ]; then
    echo "ERROR: Instance did not become ready in time"
    exit 1
fi

# Wait a bit more for SSH to be ready
echo "Waiting for SSH to be available..."
sleep 30

# Run setup and benchmarks on instance
echo ""
echo "[4/7] Running setup on instance..."
ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'REMOTE_SCRIPT'
    set -e

    # Clone repository
    git clone https://github.com/ALJainProjects/longlive-optimization.git ~/longlive-optimization
    cd ~/longlive-optimization

    # Run setup
    bash scripts/setup_h100.sh

    # Activate environment
    source ~/longlive-env/bin/activate

    echo "Setup complete!"
REMOTE_SCRIPT

echo ""
echo "[5/7] Running benchmarks..."
ssh ubuntu@$INSTANCE_IP << 'REMOTE_SCRIPT'
    source ~/longlive-env/bin/activate
    cd ~/longlive-optimization

    # Run quick benchmark first
    python benchmarks/run_benchmark.py --config quick

    # Run full benchmark
    python scripts/run_full_benchmark.py

    # Generate comparison videos (3 prompts, 60 frames for speed)
    python scripts/generate_comparison_videos.py \
        --output-dir comparison_videos \
        --num-frames 60 \
        --num-prompts 3

    echo "Benchmarks complete!"
REMOTE_SCRIPT

# Download results
echo ""
echo "[6/7] Downloading results..."
mkdir -p "$LOCAL_RESULTS_DIR"

scp -r ubuntu@$INSTANCE_IP:~/longlive-optimization/benchmark_results "$LOCAL_RESULTS_DIR/"
scp -r ubuntu@$INSTANCE_IP:~/longlive-optimization/comparison_videos "$LOCAL_RESULTS_DIR/"

echo "Results downloaded to: $LOCAL_RESULTS_DIR"

# Terminate instance
echo ""
echo "[7/7] Terminating instance..."
TERMINATE_RESPONSE=$(curl -s -u "$LAMBDA_API_KEY:" \
    -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}")

echo "$TERMINATE_RESPONSE"

echo ""
echo "=============================================="
echo "COMPLETE!"
echo "=============================================="
echo ""
echo "Results are in: $LOCAL_RESULTS_DIR"
echo ""
echo "View comparison videos:"
echo "  open $LOCAL_RESULTS_DIR/comparison_videos/comparison.html"
echo ""
echo "View benchmark results:"
echo "  open $LOCAL_RESULTS_DIR/benchmark_results/comparison_*.html"
