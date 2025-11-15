# Training Your Chess Bot on Modal

This guide explains how to train an improved chess neural network using Modal's serverless GPU platform.

## Why Modal?

- **GPU Access**: Train on powerful GPUs (A10G, A100) without local hardware
- **Cost-Effective**: Pay only for compute time used (~$0.50-$2 for typical training)
- **No Setup**: No local GPU drivers or CUDA installation needed
- **Scalable**: Easy to run multiple experiments in parallel

## Prerequisites

1. **Install Modal**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal setup
   ```
   This will open a browser window to create a free Modal account and get your API token.

## Training on Modal

### Basic Usage

Run the training script:

```bash
modal run modal_train.py
```

This will:
- Spin up a GPU instance on Modal
- Generate 5,000 training games
- Train for 30 epochs with the improved CNN architecture
- Save the best model to a Modal volume
- Automatically shut down when complete

### Monitor Progress

The training script outputs real-time progress:
- Data generation progress
- Training/validation loss per epoch
- Time per epoch
- GPU utilization

### Estimated Time & Cost

| Config | Games | Epochs | GPU | Time | Cost |
|--------|-------|--------|-----|------|------|
| Small | 2,000 | 20 | A10G | ~5 min | ~$0.10 |
| Medium | 5,000 | 30 | A10G | ~15 min | ~$0.30 |
| Large | 10,000 | 50 | A10G | ~45 min | ~$0.80 |
| X-Large | 20,000 | 100 | A100 | ~60 min | ~$4.00 |

## Customizing Training

Edit the `main()` function in `modal_train.py` to adjust parameters:

```python
best_loss = train_model.remote(
    num_games=10000,    # More games = better training data
    epochs=50,          # More epochs = better convergence
    batch_size=512,     # Larger batch = faster on GPU (if it fits)
    lr=0.001,           # Learning rate
    hidden_size=512,    # Model capacity
    val_split=0.2,      # Validation split
)
```

## Download Your Trained Model

After training completes, download the model:

```bash
modal volume get chess-models chess_model_best.pth ./chess_model.pth
```

This downloads the best model (lowest validation loss) to your local directory.

## Using the Trained Model

The model is automatically compatible with your bot! Just make sure `chess_model.pth` is in your project root, and the bot will load it.

To use the improved architecture, update `src/main.py` line 35:

```python
from .utils.improved_model import ImprovedChessModel
_model = ImprovedChessModel(hidden_size=checkpoint.get('hidden_size', 512))
```

## Advanced: Using Real Chess Data

For even better results, train on real chess games from Lichess or Chess.com:

1. Download PGN files from https://database.lichess.org/
2. Parse games and extract positions
3. Use Stockfish to evaluate positions for better labels
4. Modify `generate_training_data()` in `modal_train.py` to load from PGN

Example implementation coming soon!

## Troubleshooting

### "Module not found: modal"
```bash
pip install modal
```

### "Authentication required"
```bash
modal setup
```

### Model won't load locally
Make sure you've updated `src/main.py` to import `ImprovedChessModel` instead of `ChessModel`.

### Out of GPU memory
Reduce `batch_size` in the training parameters.

### Training is slow
- Increase batch size if GPU memory allows
- Use A100 GPU instead of A10G (change in `modal_train.py`)

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, hidden sizes
2. **Generate more data**: Increase `num_games` for better training
3. **Add data augmentation**: Flip/rotate boards for more diverse training
4. **Use real games**: Download Lichess database for professional-level play
5. **Implement self-play**: Have the bot play against itself to improve

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [Lichess Database](https://database.lichess.org/)
- [Chess Programming Wiki](https://www.chessprogramming.org/)

