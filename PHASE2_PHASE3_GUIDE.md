# Phase 2 & Phase 3 Training Guide

Complete guide for improving your chess bot with self-play reinforcement learning (Phase 2) and tactical puzzle training (Phase 3).

## Overview

Your training pipeline now has **4 phases**:

1. **Phase 1** (✅ Complete): Supervised learning on Lichess games → `chess_model_best.pth`
2. **Phase 2** (🆕 Available): Self-play reinforcement learning → `chess_model_phase2.pth`
3. **Phase 3** (🆕 Available): Tactical puzzle training → `chess_model_phase3.pth`
4. **Phase 4** (Optional): Final Modal GPU training → production model

## Quick Start

### Run Complete Pipeline (Easiest)

```bash
cd /Users/emaadqazi/Desktop/Coding\ Projects/chess-bot

# Run all phases (Phase 2 + Phase 3)
python training/pipeline.py --gpu

# Run only Phase 2 (self-play)
python training/pipeline.py --phase 2 --gpu

# Run only Phase 3 (tactical)
python training/pipeline.py --phase 3 --gpu
```

### Custom Configuration

```bash
# More self-play games for stronger training
python training/pipeline.py \
  --selfplay-games 5000 \
  --selfplay-sims 200 \
  --selfplay-epochs 150 \
  --tactical-puzzles 100000 \
  --tactical-epochs 100 \
  --gpu
```

## Phase 2: Self-Play Reinforcement Learning

### What It Does
- Bot plays against itself using improved MCTS
- Learns from its own games (reinforcement learning)
- Improves strategic understanding and position evaluation

### Manual Execution

#### Step 1: Generate Self-Play Data

```bash
# Local CPU (slow but free)
python -c "
from training.selfplay_generator import SelfPlayDataGenerator
from src.utils.improved_model import ImprovedChessModel
from src.utils.move_mapper import MoveMapper
import torch

checkpoint = torch.load('chess_model_best.pth', map_location='cpu')
model = ImprovedChessModel(hidden_size=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

generator = SelfPlayDataGenerator(model, MoveMapper(), num_simulations=100)
generator.generate_games(num_games=1000, output_path='data/selfplay.npz')
"
```

**OR Modal GPU (fast!):**

```bash
# Upload model to Modal first
modal volume put chess-models chess_model_best.pth chess_model_best.pth

# Generate games on GPU
modal run training/modal_selfplay.py --num-games 1000 --num-simulations 100

# Download results
modal volume get chess-models selfplay_batch.npz ./data/selfplay.npz
```

#### Step 2: Train on Self-Play Data

```bash
python training/train_selfplay.py \
  --model chess_model_best.pth \
  --selfplay-data data/selfplay.npz \
  --output models/chess_model_phase2.pth \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.0001 \
  --device cpu  # or cuda
```

### Expected Improvements
- **+50-100 ELO** improvement
- Better strategic play
- Improved position evaluation
- More confident move selection

## Phase 3: Tactical Puzzle Training

### What It Does
- Downloads 50K+ tactical puzzles from Lichess
- Trains model to find tactical shots (forks, pins, skewers, etc.)
- Dramatically improves tactical awareness

### Manual Execution

#### Step 1: Download Tactical Puzzles

```bash
python training/tactical_puzzles.py \
  --num-puzzles 50000 \
  --min-rating 1500 \
  --max-rating 2500 \
  --output data/tactical_puzzles.npz
```

**Download Details:**
- Downloads from Lichess puzzle database (~200MB)
- Filters by rating range
- Processes puzzles for training
- Output: ~50-100MB .npz file

#### Step 2: Train on Tactical Puzzles

```bash
python training/train_tactical.py \
  --model models/chess_model_phase2.pth \  # or chess_model_best.pth if skipping Phase 2
  --tactical-data data/tactical_puzzles.npz \
  --output models/chess_model_phase3.pth \
  --epochs 50 \
  --batch-size 512 \
  --lr 0.0001 \
  --policy-weight 2.0 \  # Emphasizes tactical accuracy
  --device cpu  # or cuda
```

### Expected Improvements
- **+100-150 ELO** improvement
- Near-perfect mate-in-1 detection
- Finds mate-in-2/3 reliably
- Catches tactical blunders instantly
- Better capture evaluation

## Timeline & Resource Usage

### Local CPU

| Phase | Step | Time | Resources |
|-------|------|------|-----------|
| Phase 2 | Generate 1K games | 3-5 hours | High CPU |
| Phase 2 | Train (100 epochs) | 1-2 hours | Moderate CPU |
| Phase 3 | Download puzzles | 5-10 min | Internet |
| Phase 3 | Train (50 epochs) | 30-60 min | Moderate CPU |

**Total (CPU):** ~5-8 hours for complete pipeline

### Modal GPU (T4)

| Phase | Step | Time | Cost |
|-------|------|------|------|
| Phase 2 | Generate 1K games | 15-20 min | ~$0.10 |
| Phase 2 | Train (100 epochs) | 20-30 min | ~$0.15 |
| Phase 3 | Download puzzles | 5-10 min | Free (local) |
| Phase 3 | Train (50 epochs) | 15-20 min | ~$0.10 |

**Total (GPU):** ~1 hour, ~$0.35

## File Structure

```
chess-bot/
├── chess_model_best.pth           # Phase 1 (current)
├── models/
│   ├── chess_model_phase2.pth     # After self-play training
│   └── chess_model_phase3.pth     # Final model with tactics
├── data/
│   ├── selfplay.npz               # Self-play games
│   └── tactical_puzzles.npz       # Tactical puzzles
└── training/
    ├── pipeline.py                # 🆕 Complete pipeline
    ├── selfplay_generator.py      # 🆕 Generate self-play data
    ├── train_selfplay.py          # 🆕 Phase 2 training
    ├── tactical_puzzles.py        # 🆕 Download puzzles
    ├── train_tactical.py          # 🆕 Phase 3 training
    └── modal_selfplay.py          # 🆕 GPU self-play
```

## Using The New Model

After Phase 2 or Phase 3 completes:

```bash
# Replace current model with Phase 3 (recommended)
cp models/chess_model_phase3.pth chess_model_best.pth

# Or use Phase 2 only
cp models/chess_model_phase2.pth chess_model_best.pth

# Test the new model
python src/main.py
```

## Configuration Options

### Pipeline Options

```bash
python training/pipeline.py --help

Options:
  --phase {all,2,3}              Which phase(s) to run
  --gpu                          Use GPU if available
  --selfplay-games N             Number of self-play games (default: 1000)
  --selfplay-sims N              MCTS simulations per move (default: 100)
  --selfplay-epochs N            Training epochs for Phase 2 (default: 100)
  --tactical-puzzles N           Number of puzzles (default: 50000)
  --tactical-epochs N            Training epochs for Phase 3 (default: 50)
```

### Recommended Configurations

**Fast Testing (30 min):**
```bash
python training/pipeline.py \
  --selfplay-games 100 \
  --selfplay-sims 50 \
  --selfplay-epochs 20 \
  --tactical-puzzles 5000 \
  --tactical-epochs 10
```

**Balanced (2-3 hours CPU, 1 hour GPU):**
```bash
python training/pipeline.py \
  --selfplay-games 1000 \
  --selfplay-sims 100 \
  --selfplay-epochs 100 \
  --tactical-puzzles 50000 \
  --tactical-epochs 50 \
  --gpu
```

**Maximum Strength (8-12 hours CPU, 3-4 hours GPU):**
```bash
python training/pipeline.py \
  --selfplay-games 5000 \
  --selfplay-sims 200 \
  --selfplay-epochs 200 \
  --tactical-puzzles 100000 \
  --tactical-epochs 100 \
  --gpu
```

## Expected Performance Gains

### Cumulative ELO Improvements

| Configuration | Phase 2 Gain | Phase 3 Gain | Total Gain |
|--------------|--------------|--------------|------------|
| Fast Testing | +20-30 | +50-80 | +70-110 |
| Balanced | +50-100 | +100-150 | +150-250 |
| Maximum | +80-120 | +120-180 | +200-300 |

### Starting ELO: ~1300-1400 (Phase 1 only)
### After Balanced Pipeline: **~1500-1650**
### After Maximum Pipeline: **~1600-1750**

## Verification & Testing

### Test Tactical Improvement

```python
from src.utils.minimax import MinimaxSearch
from src.utils.improved_model import ImprovedChessModel
from src.utils.move_mapper import MoveMapper
import chess, torch

# Load Phase 3 model
checkpoint = torch.load('models/chess_model_phase3.pth', map_location='cpu')
model = ImprovedChessModel(hidden_size=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test mate-in-1
board = chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1')
minimax = MinimaxSearch(model, MoveMapper(), depth=3)
best_move, _ = minimax.search(board)

print(f"Best move: {best_move.uci()}")  # Should be Qxf7# (checkmate!)
```

### Compare Models

```bash
# Benchmark Phase 1 vs Phase 3
python benchmarks/benchmark_stockfish.py --model chess_model_best.pth
python benchmarks/benchmark_stockfish.py --model models/chess_model_phase3.pth
```

## Troubleshooting

### Self-Play Generation Too Slow
- Use Modal GPU: `modal run training/modal_selfplay.py`
- Reduce `--selfplay-games` to 500 or 100
- Reduce `--selfplay-sims` to 50

### Tactical Download Fails
- The script will fallback to sample puzzles
- Check internet connection
- Try smaller `--num-puzzles` (e.g., 10000)

### Out of Memory
- Reduce `--batch-size` (try 128 or 64)
- Use CPU instead of GPU
- Close other applications

### Training Not Improving
- Ensure input model exists
- Check data was generated correctly
- Try lower learning rate (e.g., 0.00005)
- Increase epochs

## Next Steps

After completing Phase 2 & 3:

1. **Test Your Bot**: Play games and verify improvements
2. **Benchmark**: Compare against Stockfish at different levels
3. **Iterate**: Run pipeline again with more games/puzzles
4. **Share**: Your model is now strong enough for tournaments!

## Advanced: Iterative Self-Play

For maximum strength, run multiple self-play cycles:

```bash
# Cycle 1: Phase 2 & 3
python training/pipeline.py --gpu

# Use Phase 3 as new baseline
cp models/chess_model_phase3.pth chess_model_best.pth

# Cycle 2: More self-play with improved model
python training/pipeline.py --phase 2 --selfplay-games 2000 --gpu
cp models/chess_model_phase2.pth chess_model_best.pth

# Cycle 3: More tactics
python training/pipeline.py --phase 3 --tactical-puzzles 100000 --gpu

# Final model
cp models/chess_model_phase3.pth chess_model_best_final.pth
```

Each cycle builds on previous improvements!

---

**Questions?** Check the inline documentation in each script or open an issue.

**Ready to train?** Start with:
```bash
python training/pipeline.py --gpu
```
