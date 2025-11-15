# 🚀 Benchmark Quick Start

## Setup (One-Time)

```bash
# 1. Download and install Stockfish
bash download_stockfish.sh

# 2. Verify it works
stockfish
# (type 'quit' to exit)
```

## Run Your First Benchmark

```bash
# Activate Python environment
source .venv/bin/activate

# Run 10 games against 1200 ELO
python benchmark_stockfish.py --elo 1200 --games 10
```

**Takes ~5-10 minutes**

## What You'll See

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
Wins:   3 (30.0%)
Losses: 6 (60.0%)
Draws:  1 (10.0%)

Score: 35.0% (vs 1200 ELO)
==================================================================

📊 Estimated Bot ELO: ~1050

✓ Results saved to: benchmarks/stockfish_benchmark_20241115_123456.json
```

## Find Your Bot's True Strength

```bash
# Progressive test (highly recommended!)
python benchmark_stockfish.py --progressive
```

This will:
- Start at 800 ELO
- Gradually increase difficulty
- Stop when your bot struggles
- Give accurate ELO estimate

**Takes ~20-30 minutes**

## Common Commands

```bash
# Quick test against beginner (800 ELO)
python benchmark_stockfish.py --elo 800 --games 10

# Test against club player (1600 ELO)  
python benchmark_stockfish.py --elo 1600 --games 10

# Stronger bot (more thinking time)
python benchmark_stockfish.py --elo 1200 --games 10 --mcts-sims 100

# Faster bot (less thinking time)
python benchmark_stockfish.py --elo 1200 --games 10 --mcts-sims 25
```

## Expected Results (Current Model)

Your bot was trained on **random games**, so expect:

- **vs 800 ELO**: ~50-70% score (should win/draw)
- **vs 1000 ELO**: ~40-50% score (competitive)  
- **vs 1200 ELO**: ~25-35% score (struggling)
- **Estimated ELO**: ~900-1100

## After Training with Real Games

Once you retrain with Lichess games:

- **vs 1200 ELO**: ~60-75% score  
- **vs 1400 ELO**: ~50-60% score
- **vs 1600 ELO**: ~35-45% score
- **Estimated ELO**: ~1400-1600

**That's a ~400 point improvement!** 📈

## Tracking Improvement

```bash
# Before new training
python benchmark_stockfish.py --progressive
# Note: Estimated ELO ~1050

# Train new model
modal run modal_train.py
modal volume get chess-models chess_model_best.pth ./chess_model.pth --force

# After new training  
python benchmark_stockfish.py --progressive
# Note: Estimated ELO ~1420 (+370 points!)
```

## View Past Results

```bash
# List all benchmarks
ls -lt benchmarks/

# View specific result
cat benchmarks/stockfish_benchmark_20241115_123456.json | python -m json.tool
```

## Need Help?

- **Full guide**: See `BENCHMARKING.md`
- **Stockfish issues**: See `STOCKFISH_SETUP.md`
- **Training improvements**: See `TRAINING_GUIDE.md`

---

**Ready to see how strong your bot is? Run the benchmark! 🎯**

