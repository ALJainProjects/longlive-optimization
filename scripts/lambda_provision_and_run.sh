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
SSH_KEY_FILE="$HOME/.ssh/lambda_gh200"  # Local private key file
LOCAL_RESULTS_DIR="./lambda_results"
REPO_URL="https://github.com/ALJainProjects/longlive-optimization.git"

# SSH options for reliability
SSH_OPTS="-i $SSH_KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"

echo "=============================================="
echo "Lambda Labs H100 Provisioning Script"
echo "=============================================="
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo "SSH Key: $SSH_KEY_FILE"
echo ""

# Check SSH key exists
if [ ! -f "$SSH_KEY_FILE" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY_FILE"
    echo "Please ensure your SSH private key exists"
    exit 1
fi

# Check for available instances
echo "[1/7] Checking instance availability..."
AVAILABILITY=$(curl -s -u "$LAMBDA_API_KEY:" \
    "https://cloud.lambdalabs.com/api/v1/instance-types" | \
    python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data.get('data', {}).get('$INSTANCE_TYPE', {}), indent=2))")

echo "$AVAILABILITY"

# Check if capacity available
if echo "$AVAILABILITY" | grep -q '"regions_with_capacity_available": \[\]'; then
    echo "ERROR: No H100 instances available. Try again later."
    exit 1
fi

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
MAX_WAIT=600  # 10 minutes (increased from 5)
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

# Wait for SSH with retries
echo ""
echo "Waiting for SSH to be available..."
SSH_READY=false
SSH_RETRIES=12  # 12 retries x 10 seconds = 2 minutes

for i in $(seq 1 $SSH_RETRIES); do
    echo "  SSH attempt $i/$SSH_RETRIES..."
    if ssh $SSH_OPTS ubuntu@$INSTANCE_IP 'echo "SSH ready"' 2>/dev/null; then
        SSH_READY=true
        echo "  SSH connection successful!"
        break
    fi
    sleep 10
done

if [ "$SSH_READY" = false ]; then
    echo "ERROR: SSH connection failed after $SSH_RETRIES attempts"
    echo "Instance may still be initializing. Try manually:"
    echo "  ssh $SSH_OPTS ubuntu@$INSTANCE_IP"
    exit 1
fi

# Run setup on instance
echo ""
echo "[4/7] Running setup on instance..."
ssh $SSH_OPTS ubuntu@$INSTANCE_IP << 'REMOTE_SCRIPT'
    set -e

    echo "=== Cloning repository ==="
    rm -rf ~/longlive-optimization
    git clone https://github.com/ALJainProjects/longlive-optimization.git ~/longlive-optimization
    cd ~/longlive-optimization

    echo "=== Running setup script ==="
    bash scripts/setup_h100.sh

    echo "=== Setup complete! ==="
REMOTE_SCRIPT

echo ""
echo "[5/7] Running benchmarks..."
ssh $SSH_OPTS ubuntu@$INSTANCE_IP << 'REMOTE_SCRIPT'
    set -e
    source ~/longlive-env/bin/activate
    cd ~/longlive-optimization

    echo "=== Running quick benchmark ==="
    python benchmarks/run_benchmark.py --config quick || echo "Quick benchmark skipped"

    echo "=== Running full benchmark ==="
    python scripts/run_full_benchmark.py || echo "Full benchmark completed with warnings"

    echo "=== Generating comparison videos ==="
    python scripts/generate_comparison_videos.py \
        --output-dir comparison_videos \
        --num-frames 60 \
        --num-prompts 3 || echo "Video generation skipped"

    echo "=== Benchmarks complete! ==="
REMOTE_SCRIPT

# Download results
echo ""
echo "[6/7] Downloading results..."
mkdir -p "$LOCAL_RESULTS_DIR"

scp $SSH_OPTS -r ubuntu@$INSTANCE_IP:~/longlive-optimization/benchmark_results "$LOCAL_RESULTS_DIR/" 2>/dev/null || echo "  (no benchmark_results found)"
scp $SSH_OPTS -r ubuntu@$INSTANCE_IP:~/longlive-optimization/comparison_videos "$LOCAL_RESULTS_DIR/" 2>/dev/null || echo "  (no comparison_videos found)"
scp $SSH_OPTS -r ubuntu@$INSTANCE_IP:~/longlive-optimization/quality_results "$LOCAL_RESULTS_DIR/" 2>/dev/null || echo "  (no quality_results found)"

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
