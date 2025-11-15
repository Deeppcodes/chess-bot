# 🎯 Chess Bot Benchmarking Guide

This guide explains how to benchmark your chess bot to measure its strength and track improvements.

## Quick Start

### 1. Install Stockfish

```bash
# Run the installation script
bash install_stockfish.sh

# Or manually check if it's installed
stockfish
```

If the script doesn't work, manually download from: https://stockfishchess.org/download/

### 2. Run Your First Benchmark

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run a quick benchmark (10 games vs 1200 ELO)
python benchmark_stockfish.py --elo 1200 --games 10
```

This will:
- Play 10 games against Stockfish at 1200 ELO
- Show results after each game
- Give you an estimated ELO rating
- Save results to `benchmarks/` folder

## Benchmark Options

### Single ELO Level

Test against a specific Stockfish strength:

```bash
# Beginner (800 ELO)
python benchmark_stockfish.py --elo 800 --games 10

# Casual player (1200 ELO)
python benchmark_stockfish.py --elo 1200 --games 10

# Club player (1600 ELO)
python benchmark_stockfish.py --elo 1600 --games 10

# Advanced (2000 ELO)
python benchmark_stockfish.py --elo 2000 --games 10
```

### Progressive Benchmark (Recommended!)

Automatically tests multiple levels and finds your bot's strength:

```bash
python benchmark_stockfish.py --progressive
```

This will:
- Start at 800 ELO and go up
- Stop when your bot starts struggling
- Estimate your bot's ELO rating
- Takes ~20-30 minutes

### Adjust Bot Strength

Control how much your bot "thinks":

```bash
# Faster but weaker (25 simulations)
python benchmark_stockfish.py --elo 1200 --mcts-sims 25 --games 10

# Default (50 simulations)
python benchmark_stockfish.py --elo 1200 --mcts-sims 50 --games 10

# Stronger but slower (200 simulations)
python benchmark_stockfish.py --elo 1200 --mcts-sims 200 --games 10
```

## Understanding Results

### Example Output

```
==================================================================
Benchmark: 10 games vs Stockfish 1200 ELO
==================================================================

Game 1/10:
  White: Bot, Black: Stockfish
  ✓ Bot wins! (45.3s)

Game 2/10:
  White: Stockfish, Black: Bot
  ✗ Bot loses (32.1s)

...

==================================================================
Results Summary
==================================================================
Games Played: 10
Wins:   4 (40.0%)
Losses: 5 (50.0%)
Draws:  1 (10.0%)

Score: 45.0% (vs 1200 ELO)
==================================================================

📊 Estimated Bot ELO: ~1150

✓ Results saved to: benchmarks/stockfish_benchmark_20241115_123456.json
```

### What Does This Mean?

- **Score**: Percentage of points earned (win = 1 point, draw = 0.5, loss = 0)
- **~50% score** = Bot is roughly equal to that ELO level
- **>70% score** = Bot is stronger than that level
- **<30% score** = Bot is weaker than that level

### ELO Reference

| ELO Range | Level | Typical Players |
|-----------|-------|-----------------|
| 800-1000 | Beginner | Just learned rules |
| 1000-1200 | Novice | Casual players |
| 1200-1400 | Intermediate | Regular club members |
| 1400-1600 | Club Player | Strong club players |
| 1600-1800 | Advanced | Tournament players |
| 1800-2000 | Expert | Strong tournament players |
| 2000-2200 | Master | National Master level |
| 2200+ | Grandmaster | Elite players |

## Tracking Improvement

### Compare Benchmark Results

Each benchmark saves a JSON file with full details:

```bash
# List your benchmarks
ls -lt benchmarks/

# View a specific benchmark
cat benchmarks/stockfish_benchmark_20241115_123456.json
```

### Benchmark Before and After Training

```bash
# 1. Benchmark current model
python benchmark_stockfish.py --progressive
# Note the ELO: ~1100

# 2. Train new model with better data
modal run modal_train.py
modal volume get chess-models chess_model_best.pth ./chess_model.pth --force

# 3. Benchmark again
python benchmark_stockfish.py --progressive
# New ELO: ~1400 (300 points improvement!)
```

## Expected Performance

Based on training data quality:

| Training Data | Expected ELO | Benchmark Score vs 1200 |
|---------------|--------------|-------------------------|
| Random games (current) | 900-1100 | 25-35% |
| Lichess games | 1300-1500 | 55-70% |
| Lichess + more training | 1500-1700 | 65-80% |
| Lichess + self-play | 1700-2000+ | 75%+ |

## Tips for Better Benchmarking

### 1. Use Consistent Settings

Always use the same MCTS simulations for fair comparison:

```bash
# Always use 50 simulations for benchmarks
python benchmark_stockfish.py --elo 1200 --mcts-sims 50
```

### 2. Play Enough Games

- **Minimum: 10 games** for rough estimate
- **Recommended: 20 games** for decent accuracy  
- **Best: 50+ games** for statistical significance

### 3. Test Multiple Levels

Don't just test one ELO level:

```bash
# Test range to find your level
python benchmark_stockfish.py --elo 800 --games 10
python benchmark_stockfish.py --elo 1000 --games 10
python benchmark_stockfish.py --elo 1200 --games 10
python benchmark_stockfish.py --elo 1400 --games 10
```

Or use progressive mode:

```bash
python benchmark_stockfish.py --progressive
```

## Troubleshooting

### "Stockfish not found"

Install Stockfish:

```bash
bash install_stockfish.sh
```

Or manually:
- **macOS**: `brew install stockfish`
- **Ubuntu**: `sudo apt-get install stockfish`  
- **Windows**: Download from https://stockfishchess.org/download/

### "Model not loading"

Make sure you have a trained model:

```bash
ls -lh chess_model.pth
```

If not, train one:

```bash
modal run modal_train.py
modal volume get chess-models chess_model_best.pth ./chess_model.pth
```

### Games are too slow

Reduce MCTS simulations:

```bash
python benchmark_stockfish.py --elo 1200 --mcts-sims 25
```

### Bot plays too weak

Your model needs better training data! See:
- `TRAINING_GUIDE.md` for training with real games
- Ask me to add Lichess games to your training

## Advanced: Custom Benchmarks

Edit `benchmark_stockfish.py` to:
- Test specific positions
- Analyze move quality
- Compare different models
- Create custom metrics

## Next Steps

1. ✅ Run your first benchmark
2. ✅ Note your current ELO
3. ✅ Train with better data (Lichess games)
4. ✅ Re-benchmark to see improvement
5. ✅ Iterate and improve!

---

**Happy benchmarking! 📊**

