#!/bin/bash
# Poll for H100 capacity and launch when available

set -e

if [ -f ~/.ray-cluster-credentials ]; then
    source ~/.ray-cluster-credentials
fi

if [ -z "$LAMBDA_API_KEY" ]; then
    echo "ERROR: LAMBDA_API_KEY not set"
    exit 1
fi

API_URL="https://cloud.lambdalabs.com/api/v1"

echo "Polling for H100 capacity..."
while true; do
    result=$(curl -s -X POST "$API_URL/instance-operations/launch" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"region_name": "us-west-1", "instance_type_name": "gpu_1x_h100_pcie", "ssh_key_names": ["lambda-gh200"], "quantity": 1, "name": "longlive-benchmark"}')

    if echo "$result" | grep -q "instance_ids"; then
        echo ""
        echo "SUCCESS at $(date):"
        echo "$result"
        instance_id=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['instance_ids'][0])")
        echo "Instance ID: $instance_id"

        echo ""
        echo "Waiting 60s for instance to boot..."
        sleep 60

        # Get IP
        ip=$(curl -s -X GET "$API_URL/instances/$instance_id" \
            -H "Authorization: Bearer $LAMBDA_API_KEY" \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['ip'] or 'pending')")

        echo "Instance IP: $ip"
        echo ""
        echo "To setup and run benchmarks:"
        echo "  ssh -i ~/.ssh/lambda_gh200 ubuntu@$ip 'git clone https://github.com/ALJainProjects/longlive-optimization.git ~/longlive-optimization && cd ~/longlive-optimization && bash scripts/setup_h100.sh --run-benchmark'"
        break
    fi

    echo "$(date): No H100 capacity, retrying in 60s..."
    sleep 60
done
