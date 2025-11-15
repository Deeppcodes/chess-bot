# 🚀 Training Scripts

This directory contains all scripts for training your chess bot.

## 📁 Files

- **`modal_train.py`** - Train on Modal GPU (recommended, fast)
- **`train.py`** - Train locally on CPU (slower, for testing)
- **`generate_data.py`** - Generate training data locally
- **`modal_requirements.txt`** - Modal-specific dependencies

## 🎯 Quick Start

### Modal Training (Recommended)

```bash
# From project root
modal run training/modal_train.py

# Download trained model
modal volume get chess-models chess_model_best.pth ./models/chess_model.pth
```

### Local Training

```bash
# Generate data
python training/generate_data.py --num-games 1000

# Train model
python training/train.py --data training_data.npz --epochs 20
```

## 📊 Training Options

### Modal Training (GPU)

**Advantages:**
- ✅ Fast (15-30 minutes)
- ✅ Powerful GPU
- ✅ No local GPU needed
- ✅ Cost-effective (~$0.30)

**Usage:**
```bash
modal run training/modal_train.py
```

Edit `training/modal_train.py` to customize:
- Number of games
- Epochs
- Batch size
- Learning rate
- Model architecture

### Local Training (CPU)

**Advantages:**
- ✅ Free
- ✅ No internet needed
- ✅ Full control

**Disadvantages:**
- ❌ Slow (hours)
- ❌ Limited by CPU

**Usage:**
```bash
python training/train.py --epochs 20 --batch-size 32
```

## 📈 Expected Results

| Training Data | Time | Cost | Bot ELO |
|----------------|------|------|---------|
| Random games | 15 min | $0.30 | ~900-1100 |
| Lichess games | 20 min | $0.40 | ~1400-1600 |
| Lichess + more | 45 min | $0.80 | ~1700-1900 |

## 🔧 Customization

### Modal Training Parameters

Edit `training/modal_train.py`:

```python
best_loss = train_model.remote(
    num_games=10000,    # More games
    epochs=50,          # More epochs
    batch_size=512,     # Larger batch
    lr=0.001,           # Learning rate
    hidden_size=512,    # Model size
)
```

### Local Training Parameters

```bash
python training/train.py \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-size 256
```

## 📚 Documentation

- **Complete Guide**: See `docs/TRAINING_GUIDE.md`
- **Modal Details**: See `docs/MODAL_TRAINING.md`
- **Quick Reference**: See `docs/QUICK_REFERENCE.md`

## 💡 Tips

1. **Start small**: Test with 1000 games first
2. **Monitor loss**: Should decrease over epochs
3. **Use real data**: Lichess games are much better
4. **Iterate**: Train → Benchmark → Improve → Repeat

## 🎯 Next Steps

After training:
1. Download model: `modal volume get chess-models chess_model_best.pth ./models/chess_model.pth`
2. Benchmark: `python benchmarks/benchmark_stockfish.py --progressive`
3. Test: `cd devtools && npm run dev`

