#!/bin/bash
# Monitor the progress of loss weight testing

echo "=========================================="
echo "MONITORING LOSS WEIGHT TESTING"
echo "=========================================="
echo ""

# Find the most recent test directory
TEST_DIR=$(ls -td results/loss_weight_tests_* 2>/dev/null | head -1)

if [ -z "$TEST_DIR" ]; then
    echo "⚠️  No test directory found yet. Testing may not have started."
    echo ""
    echo "Check if testing is running:"
    ps aux | grep -E "test_loss_configurations" | grep -v grep || echo "  No test process found"
    exit 1
fi

echo "Test directory: $TEST_DIR"
echo ""

# List completed configurations
echo "Completed configurations:"
for dir in "$TEST_DIR"/*/; do
    if [ -d "$dir" ]; then
        config_name=$(basename "$dir")
        if [ -f "$dir/metrics.json" ]; then
            echo "  ✓ $config_name (metrics available)"
        elif [ -f "$dir/test.log" ]; then
            echo "  ⏳ $config_name (in progress or completed, no metrics yet)"
        else
            echo "  ⏳ $config_name (starting...)"
        fi
    fi
done

echo ""

# Show latest metrics if available
LATEST_METRICS=$(find "$TEST_DIR" -name "metrics.json" | sort | tail -1)
if [ -n "$LATEST_METRICS" ]; then
    echo "Latest completed test metrics:"
    python << PYTHON_SCRIPT
import json
from pathlib import Path

with open('${LATEST_METRICS}', 'r') as f:
    m = json.load(f)

print(f"  Quality: {m.get('quality', 0):.3f}")
print(f"  Ramachandran Favored: {m.get('rama_favored', 0):.1%}")
print(f"  Trans Fraction: {m.get('trans', 0):.1%}")
print(f"  Clash Rate: {m.get('clash', 0):.3f}")
PYTHON_SCRIPT
    echo ""
fi

# Check if testing is still running
if ps aux | grep -E "test_loss_configurations" | grep -v grep > /dev/null; then
    echo "✓ Testing is still running..."
    echo ""
    echo "To see live progress:"
    echo "  tail -f $TEST_DIR/*/test.log"
else
    echo "⚠️  Testing process not found. It may have completed or encountered an error."
    echo ""
    echo "Check logs:"
    echo "  cat $TEST_DIR/*/test.log"
fi

echo ""
echo "To view comparison table (when all tests complete):"
echo "  python -c \"import json; from pathlib import Path; [print(f\"{p.parent.name}: {json.load(open(p))}\") for p in Path('$TEST_DIR').rglob('metrics.json')]\""



