# 🔧 Stockfish Setup Guide

Stockfish is needed to benchmark your chess bot. Here's how to install it:

## Option 1: Automatic Download (Recommended)

```bash
bash download_stockfish.sh
```

This will:
- Download the correct Stockfish version for your Mac
- Install it to `/usr/local/bin/stockfish`
- Require your password for installation

## Option 2: Using Homebrew

If you have Homebrew installed:

```bash
brew install stockfish
```

## Option 3: Manual Download

1. Visit: https://stockfishchess.org/download/
2. Download "Stockfish 16.1 for macOS"
3. Choose the right version:
   - **Apple Silicon (M1/M2/M3)**: Download the ARM version
   - **Intel Mac**: Download the x86-64 version
4. Extract the downloaded file
5. Open Terminal and run:
   ```bash
   sudo cp stockfish-macos-*/stockfish-* /usr/local/bin/stockfish
   sudo chmod +x /usr/local/bin/stockfish
   ```

## Verify Installation

After installation, test it:

```bash
stockfish
```

You should see:
```
Stockfish 16.1 by the Stockfish developers
```

Type `quit` to exit.

## Now Run Benchmarks!

Once Stockfish is installed:

```bash
# Quick benchmark (10 games)
python benchmark_stockfish.py --elo 1200 --games 10

# Progressive benchmark (finds your bot's level)
python benchmark_stockfish.py --progressive
```

See `BENCHMARKING.md` for complete guide!

## Troubleshooting

### "command not found: stockfish"

Make sure `/usr/local/bin` is in your PATH:

```bash
echo $PATH
```

If not, add to your `~/.zshrc`:

```bash
export PATH="/usr/local/bin:$PATH"
```

Then restart your terminal.

### "Permission denied"

The script needs sudo access to install to `/usr/local/bin`. Enter your Mac password when prompted.

### Still having issues?

The benchmark script can use Stockfish from any location:

```bash
# Download stockfish somewhere
# Then tell the benchmark where it is:
python benchmark_stockfish.py --stockfish-path /path/to/stockfish --elo 1200
```

