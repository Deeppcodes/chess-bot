# 🎯 Benchmarking Tools

This directory contains all tools for benchmarking your chess bot against Stockfish.

## 📁 Directory Structure

```
benchmarks/
├── benchmark_stockfish.py    # Main benchmark script
├── download_stockfish.sh     # Install Stockfish locally
├── install_stockfish.sh      # Alternative installer
├── stockfish/                 # Stockfish binary (auto-installed)
│   └── stockfish
└── results/                   # Benchmark results (JSON files)
    └── *.json
```

## 🚀 Quick Start

### 1. Install Stockfish

```bash
# From project root
bash benchmarks/download_stockfish.sh
```

### 2. Run Benchmark

```bash
# From project root
python benchmarks/benchmark_stockfish.py --elo 1400 --games 10
```

### 3. Progressive Test (Recommended)

```bash
# Finds your bot's true strength
python benchmarks/benchmark_stockfish.py --progressive
```

## 📊 Usage Examples

### Quick Test (5-10 minutes)
```bash
python benchmarks/benchmark_stockfish.py --elo 1400 --games 10
```

### Full Evaluation (20-30 minutes)
```bash
python benchmarks/benchmark_stockfish.py --progressive
```

### Test Specific Level
```bash
python benchmarks/benchmark_stockfish.py --elo 1600 --games 20
```

### Adjust Bot Strength
```bash
# Faster (weaker)
python benchmarks/benchmark_stockfish.py --elo 1400 --mcts-sims 25

# Stronger (slower)
python benchmarks/benchmark_stockfish.py --elo 1400 --mcts-sims 200
```

## 📈 Understanding Results

Results are saved to `benchmarks/results/` as JSON files with:
- Win/loss/draw record
- Score percentage
- Estimated ELO rating
- Detailed game logs

## 🔧 Stockfish Location

Stockfish is installed locally in `benchmarks/stockfish/stockfish` by default.

The benchmark script automatically finds it, but you can specify a path:
```bash
python benchmarks/benchmark_stockfish.py --stockfish-path /path/to/stockfish
```

## 📚 More Information

- **Complete Guide**: See `docs/BENCHMARKING.md`
- **Quick Start**: See `docs/BENCHMARK_QUICK_START.md`
- **Setup Help**: See `docs/STOCKFISH_SETUP.md`

## 🐛 Troubleshooting

**Stockfish not found?**
```bash
bash benchmarks/download_stockfish.sh
```

**Model not loading?**
Make sure `models/chess_model.pth` exists. Train a model first!

**Games too slow?**
Reduce MCTS simulations: `--mcts-sims 25`

