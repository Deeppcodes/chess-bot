# Quick Reference Card

## 🎯 Most Common Commands

### Training on Modal (Recommended)
```bash
# One-time setup
pip install modal
modal setup

# Train model on GPU
modal run modal_train.py

# Download trained model
modal volume get chess-models chess_model_best.pth ./chess_model.pth
```

### Benchmarking (Measure Strength)
```bash
# Setup (one-time)
bash download_stockfish.sh

# Quick benchmark (10 games)
python benchmark_stockfish.py --elo 1200 --games 10

# Find bot's true strength
python benchmark_stockfish.py --progressive
```

### Local Development
```bash
# Setup Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run devtools (includes bot)
cd devtools
npm install
npm run dev
# Open http://localhost:3000
```

### Local Training (slower)
```bash
# Generate data
python generate_data.py --num-games 1000

# Train model
python train.py --epochs 20
```

## 📁 Project Structure

```
chess-bot/
├── src/
│   ├── main.py                    # Your bot logic (edit here!)
│   └── utils/
│       ├── model.py               # Basic MLP model
│       ├── improved_model.py      # CNN model (better!)
│       ├── mcts.py                # Monte Carlo Tree Search
│       └── board_encoder.py       # Board representation
├── devtools/                      # Web interface (don't edit)
├── chess_model.pth               # Trained model (generated)
├── modal_train.py                # Modal GPU training
├── train.py                      # Local CPU training
├── serve.py                      # Backend server
└── requirements.txt              # Python dependencies
```

## 🧠 Model Files

| Model Type | Description | When to Use |
|------------|-------------|-------------|
| **ImprovedChessModel** (CNN) | Spatial awareness, residual blocks | Production, competition |
| **ChessModel** (MLP) | Simple feedforward | Testing, learning |

## 📊 Training Parameters

### Quick Settings
```python
# Fast test (5 min, ~$0.10)
num_games=2000, epochs=20

# Recommended (15 min, ~$0.30)
num_games=5000, epochs=30

# High quality (45 min, ~$0.80)
num_games=10000, epochs=50

# Maximum (2 hrs, ~$4)
num_games=20000, epochs=100
```

## 🔍 Checking Your Bot

### Is the model loaded?
Look for this in logs when running devtools:
```
✓ Loaded model from /path/to/chess_model.pth
Model parameters: 2,345,678
```

### Is it using the right model?
```
Loading ImprovedChessModel (CNN-based)    ← Good!
Loading ChessModel (MLP-based)            ← Basic model
Falling back to random moves              ← No model found
```

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| "Model not found" | Train a model or download one |
| "Random moves" | Model failed to load, check logs |
| "Out of memory" | Reduce `batch_size` in training |
| "Modal auth failed" | Run `modal setup` |
| "Devtools won't start" | Check Desktop folder permissions |
| "Import error" | Activate venv: `source .venv/bin/activate` |

## 🚀 Workflow

1. **Setup** (one time)
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   modal setup
   ```

2. **Train model**
   ```bash
   modal run modal_train.py
   modal volume get chess-models chess_model_best.pth ./chess_model.pth
   ```

3. **Test locally**
   ```bash
   cd devtools
   npm run dev
   ```

4. **Iterate**
   - Edit `src/main.py` for bot logic
   - Adjust MCTS simulations
   - Retrain with different parameters
   - Test against yourself

5. **Deploy**
   - Follow ChessHacks deployment guide
   - Your bot is ready!

## 📚 Documentation

- `README.md` - Project overview
- `TRAINING_GUIDE.md` - Complete training guide
- `MODAL_TRAINING.md` - Detailed Modal instructions
- `QUICK_REFERENCE.md` - This file!

## 💡 Tips

- **Start simple**: Use default settings first
- **Monitor training**: Watch for decreasing loss
- **Test often**: Use devtools after every change
- **Experiment**: Try different MCTS simulations
- **Use real data**: Lichess games are much better
- **Version models**: Name files like `chess_model_v1.pth`

## 🎮 Making Your Bot Smarter

1. **Better training data**: Use Lichess games
2. **More simulations**: Increase MCTS count
3. **Larger model**: Increase `hidden_size`
4. **More training**: Increase `epochs`
5. **Opening book**: Add opening move database
6. **Endgame tables**: Use tablebase for perfect endgames

---

**Need help?** Check the full docs or join [ChessHacks Discord](https://docs.chesshacks.dev/resources/discord)

