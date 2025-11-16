# 🚀 Phase 2: Self-Play Training Guide

## Overview

Phase 2 implements AlphaZero-style self-play reinforcement learning:
1. Generate self-play games using your baseline model
2. Train on the self-play data
3. Evaluate the new model vs baseline
4. Iterate if the new model is stronger

---

## 📋 Prerequisites

- ✅ Baseline model: `chess_model_best.pth`
- ✅ Virtual environment activated
- ✅ All dependencies installed

---

## 🎯 Complete Phase 2 Workflow

### **Step 1: Generate Self-Play Data**

Generate 200 games using the baseline model:

```powershell
python selfplay_generator.py --games 200 --model chess_model_best.pth --output datasets/selfplay/
```

**What this does:**
- Loads `chess_model_best.pth`
- Plays 200 games against itself using MCTS
- Saves to `datasets/selfplay/game_batch_0001.npz`
- Auto-increments batch number for subsequent runs

**Output format:**
- `boards`: (N, 12, 8, 8) - Board states
- `policies`: (N, 256) - MCTS visit count distributions
- `values`: (N,) - Game outcomes

---

### **Step 2: Train on Self-Play Data**

Train on the generated data:

```powershell
python -m src.training.train_selfplay --data-dir datasets/selfplay --model chess_model_best.pth --epochs 5 --output chess_model_sp_v1.pth
```

**What this does:**
- Loads all `.npz` files from `datasets/selfplay/`
- Loads `chess_model_best.pth` as starting point
- Trains for 5 epochs with early stopping
- Saves best model to `chess_model_sp_v1.pth`

**Optional flags:**
- `--use-amp` - Enable mixed precision for faster training (GPU only)
- `--batch-size 128` - Increase batch size if you have enough memory
- `--lr 0.0005` - Lower learning rate for fine-tuning

---

### **Step 3: Evaluate New Model**

Compare the new model against the baseline:

```powershell
python -m src.evaluation.test_model_comparison --model1 chess_model_best.pth --model2 chess_model_sp_v1.pth --games 10
```

**What this does:**
- **Policy Accuracy**: How often models agree on best move
- **Value MAE**: Difference in position evaluation
- **Head-to-Head**: 10 games between models

**Interpreting results:**
- Win rate > 55% → New model is stronger ✅
- Win rate < 45% → New model is weaker ⚠️
- Win rate ~50% → Models are similar, need more games

---

### **Step 4: Iterate (Optional)**

If the new model is stronger, use it as the new baseline:

```powershell
# Copy new model as baseline
Copy-Item chess_model_sp_v1.pth chess_model_best.pth

# Generate more data with improved model
python selfplay_generator.py --games 200 --model chess_model_best.pth --output datasets/selfplay/

# Train again
python -m src.training.train_selfplay --data-dir datasets/selfplay --model chess_model_best.pth --epochs 5 --output chess_model_sp_v2.pth

# Evaluate
python -m src.evaluation.test_model_comparison --model1 chess_model_best.pth --model2 chess_model_sp_v2.pth --games 10
```

---

## 🔥 Quick Commands

### Single File Training
```powershell
# Train on a single dataset file
python -m src.training.train_selfplay --data datasets/selfplay/game_batch_0001.npz --model chess_model_best.pth --epochs 5
```

### GPU Training with AMP
```powershell
# Faster training with mixed precision
python -m src.training.train_selfplay --data-dir datasets/selfplay --model chess_model_best.pth --epochs 5 --use-amp
```

### Quick Evaluation (Skip Some Tests)
```powershell
# Only run head-to-head games
python -m src.evaluation.test_model_comparison --model1 chess_model_best.pth --model2 chess_model_sp_v1.pth --skip-policy --skip-value --games 20
```

---

## 📊 Expected Results

After Phase 2, you should see:
- ✅ Improved policy (better move selection)
- ✅ Improved value (better position evaluation)
- ✅ Stronger gameplay (higher win rate vs baseline)

---

## 🐛 Troubleshooting

**Problem: "No .npz files found"**
- Make sure you ran `selfplay_generator.py` first
- Check that files are in `datasets/selfplay/`

**Problem: "CUDA out of memory"**
- Reduce batch size: `--batch-size 32`
- Use CPU: Remove `--use-amp` flag

**Problem: "Models play to draw every game"**
- Increase MCTS simulations: `--mcts-sims 50`
- Generate more diverse data with more games

---

## ✅ Success Criteria

Phase 2 is complete when:
1. ✅ Generated at least 200 self-play games
2. ✅ Trained successfully on self-play data
3. ✅ New model shows improvement in evaluation
4. ✅ Saved as `chess_model_sp_v1.pth`

---

**Next:** Move to Phase 3 (Tactical Training) or continue iterating Phase 2 for stronger models!

