#!/bin/bash

# Run all tests

echo "Running Parallel Universe Transformers Tests"
echo "============================================="

echo ""
echo "1. Testing SCM components..."
python tests/test_scm.py

echo ""
echo "2. Testing model components..."
python tests/test_model.py

echo ""
echo "============================================="
echo "All tests completed!"
