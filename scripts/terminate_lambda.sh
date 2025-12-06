#!/bin/bash
# Terminate Lambda Labs instances
# Usage:
#   ./scripts/terminate_lambda.sh              # Terminate all instances
#   ./scripts/terminate_lambda.sh <instance_id>  # Terminate specific instance

set -e

# Load API key
if [ -f ~/.ray-cluster-credentials ]; then
    source ~/.ray-cluster-credentials
fi

if [ -z "$LAMBDA_API_KEY" ]; then
    echo "ERROR: LAMBDA_API_KEY not set"
    echo "Set it with: export LAMBDA_API_KEY=your_key"
    exit 1
fi

API_URL="https://cloud.lambdalabs.com/api/v1"

# If instance ID provided, terminate that one
if [ -n "$1" ]; then
    echo "Terminating instance: $1"
    curl -s -X POST "$API_URL/instance-operations/terminate" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\": [\"$1\"]}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'error' in d:
    print(f'ERROR: {d[\"error\"][\"message\"]}')
else:
    for inst in d.get('data', {}).get('terminated_instances', []):
        print(f'Terminated: {inst[\"id\"]} ({inst[\"ip\"]}) - {inst[\"status\"]}')
"
    exit 0
fi

# Otherwise, list all instances and offer to terminate
echo "Fetching running instances..."
INSTANCES=$(curl -s -X GET "$API_URL/instances" \
    -H "Authorization: Bearer $LAMBDA_API_KEY")

# Check for instances
INSTANCE_COUNT=$(echo "$INSTANCES" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))")

if [ "$INSTANCE_COUNT" == "0" ]; then
    echo "No running instances found."
    exit 0
fi

echo ""
echo "Running instances:"
echo "$INSTANCES" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for inst in d.get('data', []):
    print(f'  - {inst[\"id\"]} | {inst[\"instance_type\"][\"name\"]} | {inst[\"ip\"]} | {inst[\"status\"]}')
"

echo ""
read -p "Terminate ALL instances? (y/N): " confirm

if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
    # Get all instance IDs
    IDS=$(echo "$INSTANCES" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ids = [inst['id'] for inst in d.get('data', [])]
print(json.dumps(ids))
")

    echo "Terminating all instances..."
    curl -s -X POST "$API_URL/instance-operations/terminate" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\": $IDS}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'error' in d:
    print(f'ERROR: {d[\"error\"][\"message\"]}')
else:
    for inst in d.get('data', {}).get('terminated_instances', []):
        print(f'Terminated: {inst[\"id\"]} ({inst[\"ip\"]})')
"
    echo "Done."
else
    echo "Aborted."
fi
