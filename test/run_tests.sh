#!/bin/bash

# Run all tests (run from project root: ./test/run_tests.sh)
# PYTHONPATH=code is required so imports resolve to code/

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=code

echo "Running Parallel Universe Transformers Tests"
echo "============================================="

echo ""
echo "1. Testing SCM components..."
uv run python test/test_scm.py

echo ""
echo "2. Testing model components..."
uv run python test/test_model.py

echo ""
echo "3. Checkpoint load/forward (skipped if CHECKPOINT_PATH unset)..."
uv run python test/test_checkpoint.py

echo ""
echo "============================================="
echo "All tests completed!"
