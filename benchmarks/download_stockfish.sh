#!/bin/bash
# Download and install Stockfish directly for macOS

set -e

echo "🔧 Downloading Stockfish for macOS"
echo "===================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STOCKFISH_DIR="$PROJECT_ROOT/benchmarks/stockfish"
STOCKFISH_BIN="$STOCKFISH_DIR/stockfish"

# Check if already installed locally
if [ -f "$STOCKFISH_BIN" ]; then
    echo "✓ Stockfish is already installed locally!"
    echo "Location: $STOCKFISH_BIN"
    "$STOCKFISH_BIN" -v 2>&1 | head -1
    exit 0
fi

# Check if installed system-wide
if command -v stockfish &> /dev/null; then
    echo "✓ Stockfish is already installed system-wide!"
    echo "Location: $(which stockfish)"
    stockfish -v 2>&1 | head -1
    exit 0
fi

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "Detected: Apple Silicon (M1/M2/M3)"
    URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-m1-apple-silicon.tar"
    TAR_NAME="stockfish-macos-m1-apple-silicon.tar"
else
    echo "Detected: Intel Mac"
    URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-x86-64-avx2.tar"
    TAR_NAME="stockfish-macos-x86-64-avx2.tar"
fi

echo ""
echo "Downloading Stockfish 16.1..."
echo "This may take a minute..."
echo ""

# Create temp directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Download Stockfish
curl -L -o "$TAR_NAME" "$URL"

# Extract
echo "Extracting..."
tar -xf "$TAR_NAME"

# Find the stockfish binary
STOCKFISH_BIN=$(find . -name "stockfish-*" -type f -perm +111 | head -1)

if [ -z "$STOCKFISH_BIN" ]; then
    echo "❌ Could not find stockfish binary in download"
    exit 1
fi

echo "Installing to benchmarks/stockfish/stockfish..."
echo ""

# Create stockfish directory
mkdir -p "$STOCKFISH_DIR"

# Copy and make executable
cp "$STOCKFISH_BIN" "$STOCKFISH_DIR/stockfish"
chmod +x "$STOCKFISH_DIR/stockfish"

# Clean up
cd -
rm -rf "$TMP_DIR"

echo ""
echo "✓ Stockfish installed successfully!"
echo ""
"$STOCKFISH_BIN" -v 2>&1 | head -1
echo ""
echo "Location: $STOCKFISH_BIN"
echo ""
echo "Note: Stockfish is installed locally in the project."
echo "The benchmark script will automatically find it."

