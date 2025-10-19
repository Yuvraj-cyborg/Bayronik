#!/bin/bash
# Bayronik TUI Launch Script
# Automatically sets up environment and runs the emulator

set -e  # Exit on error

echo "Bayronik"
echo "===================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Activate Python environment
echo " Activating Python environment..."
source bayronik-model/.venv/bin/activate

# Set LibTorch paths
echo "🔧 Configuring LibTorch..."
export DYLD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LIBTORCH_USE_PYTORCH=1

echo "✓ Environment ready"
echo ""

# Run the TUI
echo "🎨 Launching TUI..."
echo "   Press 'q' to quit"
echo ""

cd bayronik-infer
cargo run --release

echo ""
echo "Bayronik session ended"

