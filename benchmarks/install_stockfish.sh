#!/bin/bash
# Script to install Stockfish on macOS

set -e

echo "🔧 Stockfish Installation Script"
echo "=================================="
echo ""

# Check if Stockfish is already installed
if command -v stockfish &> /dev/null; then
    echo "✓ Stockfish is already installed!"
    echo "Location: $(which stockfish)"
    stockfish -v 2>&1 | head -1
    exit 0
fi

echo "Stockfish not found. Installing..."
echo ""

# Check if Homebrew is installed
if command -v brew &> /dev/null; then
    echo "✓ Homebrew found. Installing via Homebrew..."
    brew install stockfish
    echo ""
    echo "✓ Stockfish installed successfully!"
    stockfish -v 2>&1 | head -1
else
    echo "⚠️ Homebrew not found."
    echo ""
    echo "Please install Stockfish manually:"
    echo ""
    echo "Option 1: Install Homebrew first, then Stockfish"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  brew install stockfish"
    echo ""
    echo "Option 2: Download Stockfish directly"
    echo "  1. Visit: https://stockfishchess.org/download/"
    echo "  2. Download the macOS version"
    echo "  3. Extract and move to /usr/local/bin/"
    echo "     sudo cp stockfish /usr/local/bin/"
    echo "     sudo chmod +x /usr/local/bin/stockfish"
    echo ""
    exit 1
fi

