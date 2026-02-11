#!/bin/bash

# Run all tests (run from project root: ./test/run_tests.sh)

cd "$(dirname "$0")/.." || exit 1

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
