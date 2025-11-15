#!/bin/bash
# Quick start script for Modal training

set -e

echo "🚀 Chess Bot Modal Training - Quick Start"
echo "=========================================="
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "📦 Installing Modal..."
    pip install modal
else
    echo "✓ Modal is already installed"
fi

# Check if authenticated
if ! modal token verify &> /dev/null; then
    echo ""
    echo "🔑 Setting up Modal authentication..."
    echo "This will open your browser to create a free account."
    modal setup
else
    echo "✓ Already authenticated with Modal"
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run training: modal run modal_train.py"
echo "  2. Download model: modal volume get chess-models chess_model_best.pth ./chess_model.pth"
echo ""
echo "For more options, see MODAL_TRAINING.md"

