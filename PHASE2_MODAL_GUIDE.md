# 🚀 Phase 2: Modal GPU Workflow

## Overview

Run Phase 2 entirely on Modal GPU for **10-20x faster** execution:
- Self-play generation: ~5-10 minutes (vs 30-60 min locally)
- Training: ~5-10 minutes (vs 20-40 min locally)
- Total Phase 2: **~15-20 minutes** on Modal GPU

---

## 📋 Prerequisites

1. ✅ Modal account set up (`modal token new`)
2. ✅ Baseline model: `chess_model_best.pth` in project root
3. ✅ Modal volume created (auto-created on first run)

---

## 🎯 Complete Phase 2 Modal Workflow

### **Step 1: Generate Self-Play Data on Modal GPU**

```powershell
modal run selfplay_modal.py --games 200 --mcts-sims 50
```

**What this does:**
- Uploads `chess_model_best.pth` to Modal
- Runs 200 self-play games on A10G GPU
- Saves to Modal volume: `/data/selfplay_phase2.npz`
- Takes ~5-10 minutes

**Custom parameters:**
```powershell
# More games for better data
modal run selfplay_modal.py --games 500 --mcts-sims 50

# Faster generation (fewer MCTS sims)
modal run selfplay_modal.py --games 200 --mcts-sims 20

# Custom output name
modal run selfplay_modal.py --games 200 --output-name my_selfplay.npz
```

---

### **Step 2: Train on Modal GPU**

```powershell
modal run train_modal_selfplay.py
```

**What this does:**
- Loads `selfplay_phase2.npz` from Modal volume
- Loads `chess_model_best.pth` as baseline
- Trains for 5 epochs on A10G GPU
- Saves to Modal volume: `/data/chess_model_sp_v1.pth`
- Takes ~5-10 minutes

**Custom parameters:**
```powershell
# Train on custom dataset
modal run train_modal_selfplay.py --data-file my_selfplay.npz

# More epochs
modal run train_modal_selfplay.py --epochs 10

# Custom output name
modal run train_modal_selfplay.py --model-output chess_model_sp_v2.pth

# Larger batch size (if enough GPU memory)
modal run train_modal_selfplay.py --batch-size 512
```

---

### **Step 3: Download Trained Model**

```powershell
modal run train_modal_selfplay.py --download --model-output chess_model_sp_v1.pth
```

**What this does:**
- Downloads `chess_model_sp_v1.pth` from Modal volume to local directory
- Now you can use it locally for evaluation or devtools

---

### **Step 4: Evaluate Locally**

```powershell
python -m src.evaluation.test_model_comparison --model1 chess_model_best.pth --model2 chess_model_sp_v1.pth --games 10
```

**What this does:**
- Compares baseline vs new model
- Policy accuracy, value MAE, head-to-head games
- Determines if new model is stronger

---

## 🔄 Iterative Training Loop

If the new model is stronger, iterate:

```powershell
# 1. Copy new model as baseline
Copy-Item chess_model_sp_v1.pth chess_model_best.pth

# 2. Generate more data with improved model
modal run selfplay_modal.py --games 200

# 3. Train again
modal run train_modal_selfplay.py --model-output chess_model_sp_v2.pth

# 4. Download and evaluate
modal run train_modal_selfplay.py --download --model-output chess_model_sp_v2.pth
python -m src.evaluation.test_model_comparison --model1 chess_model_best.pth --model2 chess_model_sp_v2.pth --games 10
```

---

## 🛠️ Utility Commands

### **List files in Modal volume**
```powershell
modal run selfplay_modal.py --list-files
```

### **Download dataset from Modal**
```powershell
modal run selfplay_modal.py --download --output-name selfplay_phase2.npz
```

### **Use model from Modal volume**
```powershell
# First upload model to volume, then:
modal run selfplay_modal.py --use-volume-model --model-path chess_model_sp_v1.pth
```

---

## ⚡ Quick Start (Copy-Paste)

Run entire Phase 2 in one go:

```powershell
# Generate self-play data
modal run selfplay_modal.py --games 200 --mcts-sims 50

# Train on data
modal run train_modal_selfplay.py

# Download trained model
modal run train_modal_selfplay.py --download --model-output chess_model_sp_v1.pth

# Evaluate
python -m src.evaluation.test_model_comparison --model1 chess_model_best.pth --model2 chess_model_sp_v1.pth --games 10
```

**Total time: ~15-20 minutes** ⚡

---

## 📊 Expected Performance

| Task | Local (CPU) | Modal (A10G GPU) | Speedup |
|------|-------------|------------------|---------|
| 200 games self-play | 30-60 min | 5-10 min | **6-10x** |
| Training (5 epochs) | 20-40 min | 5-10 min | **4-8x** |
| **Total Phase 2** | **50-100 min** | **15-20 min** | **5-10x** |

---

## 🐛 Troubleshooting

**Problem: "Model not found"**
- Make sure `chess_model_best.pth` exists in project root
- Modal uploads entire project directory to `/root`

**Problem: "Data file not found"**
- Run self-play generation first
- Check files with `modal run selfplay_modal.py --list-files`

**Problem: "CUDA out of memory"**
- Reduce batch size: `--batch-size 128`
- Reduce MCTS sims: `--mcts-sims 20`

---

## ✅ Success Criteria

Phase 2 is complete when:
1. ✅ Generated 200+ self-play games on Modal
2. ✅ Trained successfully on Modal GPU
3. ✅ Downloaded `chess_model_sp_v1.pth`
4. ✅ Evaluated and confirmed improvement

---

**Next:** Continue iterating or move to Phase 3!

