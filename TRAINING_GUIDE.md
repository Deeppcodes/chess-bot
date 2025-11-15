# Chess Bot Training Guide

This guide covers all training options for your chess bot, from local training to cloud GPU training with Modal.

## Quick Comparison

| Method | Speed | Cost | GPU Required | Best For |
|--------|-------|------|--------------|----------|
| **Local Training** | Slow (CPU) | Free | No | Testing, small models |
| **Modal Training** | Fast (GPU) | ~$0.50 | No (cloud) | Production models |

---

## Option 1: Modal Training (Recommended) 🚀

Train on powerful cloud GPUs without any local GPU hardware.

### Setup (One-time, ~2 minutes)

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal setup
```

### Train Your Model

```bash
# Run training on Modal GPU
modal run modal_train.py
```

This will:
- ✅ Use a powerful GPU (A10G or A100)
- ✅ Generate training data
- ✅ Train the improved CNN model
- ✅ Save best model automatically
- ✅ Cost ~$0.30-$0.50 for typical training

Training takes **~15 minutes** and outputs progress in real-time.

### Download Your Model

After training completes:

```bash
# Download the trained model
modal volume get chess-models chess_model_best.pth ./chess_model.pth
```

That's it! Your bot will automatically load and use this model.

### Customize Training

Edit the parameters in `modal_train.py`:

```python
best_loss = train_model.remote(
    num_games=5000,     # More games = more training data
    epochs=30,          # More epochs = better training
    batch_size=256,     # Batch size (higher = faster on GPU)
    lr=0.001,           # Learning rate
    hidden_size=512,    # Model size (larger = more powerful)
)
```

**For more details**: See [MODAL_TRAINING.md](./MODAL_TRAINING.md)

---

## Option 2: Local Training

Train on your local machine (CPU only, slower).

### Generate Training Data

```bash
# Generate random games for training
python generate_data.py --num-games 1000 --output training_data.npz
```

### Train the Model

```bash
# Train on local machine
python train.py --data training_data.npz --epochs 20 --batch-size 32
```

This will:
- Use your CPU (no GPU required)
- Take longer (~30-60 minutes for small datasets)
- Save model to `chess_model.pth`

### Local Training Tips

- Start with fewer games (1000-2000) for testing
- Reduce batch size if you run out of RAM
- Training on CPU is slow but works fine for experimentation

---

## Model Architecture Comparison

### Basic Model (MLP)
- Simple feedforward neural network
- Flattens board to 768 values
- Fast but loses spatial information
- Good for: Learning, experimentation

### Improved Model (CNN) ⭐
- Convolutional neural network
- Preserves board spatial structure
- Uses residual connections
- Good for: Production, competition

The CNN model is **significantly better** and is what Modal trains by default.

---

## Training Data Quality

### Random Games (Default)
- Fast to generate
- Works for basic bot
- Limited strategic quality

### Real Games (Advanced)
Download professional games from Lichess:
1. Visit https://database.lichess.org/
2. Download PGN files (compressed)
3. Parse with `chess.pgn` module
4. Extract positions and outcomes

This produces **much better** models but requires more setup.

---

## Monitoring Training

### Modal Training
Watch real-time logs:
```
Epoch   1/30 | Train Loss: 2.3451 | Val Loss: 2.2891 | Time: 12.3s
  ✓ Saved best model (val_loss: 2.2891)
Epoch   2/30 | Train Loss: 2.1234 | Val Loss: 2.0567 | Time: 11.8s
  ✓ Saved best model (val_loss: 2.0567)
...
```

### What to Look For
- **Loss decreasing**: Good! Model is learning
- **Val loss lower than train loss**: Normal, especially with dropout
- **Val loss increasing**: Might be overfitting, reduce epochs
- **Loss not changing**: Increase learning rate or model size

---

## Using Your Trained Model

Once you have `chess_model.pth` in your project root:

1. **Test locally with devtools**:
   ```bash
   cd devtools
   npm run dev
   # Open http://localhost:3000
   ```

2. **Play against your bot** in the web interface

3. **Deploy to ChessHacks** platform when ready

---

## Troubleshooting

### "Model file not found"
Make sure `chess_model.pth` is in your project root (same directory as `train.py`).

### "Out of memory" (Local)
Reduce batch size: `python train.py --batch-size 16`

### "Modal authentication failed"
Run `modal setup` again and follow the browser prompts.

### Bot plays random moves
Check that the model is loading. Look for this line in devtools output:
```
✓ Loaded model from /path/to/chess_model.pth
```

### Training is very slow locally
This is normal on CPU. Consider using Modal for GPU training instead.

---

## Next Steps

1. ✅ **Train your first model** with Modal
2. ✅ **Test it** with the devtools interface
3. ✅ **Iterate**: Adjust hyperparameters and retrain
4. ✅ **Use real data**: Download Lichess games for better performance
5. ✅ **Deploy**: Submit to ChessHacks competition

---

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Lichess Database](https://database.lichess.org/)
- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [ChessHacks Docs](https://docs.chesshacks.dev/)

## Questions?

Check out:
- `MODAL_TRAINING.md` - Detailed Modal guide
- `README.md` - Project overview
- [ChessHacks Discord](https://docs.chesshacks.dev/resources/discord)

