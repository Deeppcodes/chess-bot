# ChessHacks Starter Bot

This is a starter bot for ChessHacks. It includes a basic bot and devtools. This is designed to help you get used to what the interface for building a bot feels like, as well as how to scaffold your own bot.

## Directory Structure

```
chess-bot/
├── src/                    # Bot source code (edit this!)
├── devtools/               # Web interface (Next.js app)
├── training/               # Training scripts (Modal + local)
├── benchmarks/             # Benchmarking tools (Stockfish)
├── docs/                   # All documentation
├── models/                 # Trained models (.pth files)
├── data/                   # Training data (.npz files)
├── scripts/                # Utility scripts
├── tests/                  # Test files
└── serve.py                # Backend server
```

**Key Directories:**

- **`/src`** - Your bot's source code. Edit `main.py` to implement your bot logic.
- **`/devtools`** - Next.js web interface for testing your bot locally.
- **`/training`** - Training scripts (`modal_train.py` for GPU training, `train.py` for local).
- **`/benchmarks`** - Benchmarking tools to measure your bot's strength against Stockfish.
- **`/docs`** - Complete documentation (guides, quick starts, troubleshooting).

`serve.py` is the backend that interacts with the Next.js app and your bot (`/src/main.py`). It handles hot reloading of your bot when you make changes. You do not need to edit this file (unless you want to).

While developing, you do not need to run either of the Python files yourself. The Next.js app includes the `serve.py` file as a subprocess, and will run it for you when you run `npm run dev`.

The backend (as a subprocess) will deploy on port `5058` by default.

This architecture is very similar to how your bot will run once you deploy it. For more information about how to deploy your bot to the ChessHacks platform, see [the docs](https://docs.chesshacks.dev/).

## Setup

Start by creating a Python virtual environment and installing the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or however you want to set up your Python.

Then, install the dependencies for the Next.js app:

```bash
cd devtools
npm install
```

Afterwards, make a copy of `.env.template` and name it `.env.local` (NOT `.env`). Then fill out the values with the path to your Python environment, and the ports you want to use.

> Copy the `.env.local` file to the `devtools` directory as well.

## Installing `devtools` (if you did not run `npx chesshacks create`)

If you started from your own project and only want to add the devtools UI, you can install it with the CLI:

```bash
npx chesshacks install
```

This will add a `devtools` folder to your current directory and ensure it is gitignored. If you want to install into a subdirectory, you can pass a path:

```bash
npx chesshacks install my-existing-bot
```

In both cases, you can then follow the instructions in [Setup](#setup) and [Running the app](#running-the-app) from inside the `devtools` folder.

## Running the app

Lastly, simply run the Nextjs app inside of the devtools folder.

```bash
cd devtools
npm run dev
```

## Troubleshooting

First, make sure that you aren't running any `python` commands! These devtools are designed to help you play against your bot and see how its predictions are working. You can see [Setup](#setup) and [Running the app](#running-the-app) above for information on how to run the app. You should be running the Next.js app, not the Python files directly!

If you get an error like this:

```python
Traceback (most recent call last):
  File "/Users/obama/dev/chesshacks//src/main.py", line 1, in <module>
    from .utils import chess_manager, GameContext
ImportError: attempted relative import with no known parent package
```

you might think that you should remove the period before `utils` and that will fix the issue. But in reality, this will just cause more problems in the future! You aren't supposed to run `main.py ` on your own—it's designed for `serve.py` to run it for you within the subprocess. Removing the period would cause it to break during that step.

### Logs

Once you run the app, you should see logs from both the Next.js app and the Python subprocess, which includes both `serve.py` and `main.py`. `stdout`s and `stderr`s from both Python files will show in your Next.js terminal. They are designed to be fairly verbose by default.

## HMR (Hot Module Reloading)

By default, the Next.js app will automatically reload (dismount and remount the subprocess) when you make changes to the code in `/src` OR press the manual reload button on the frontend. This is called HMR (Hot Module Reloading). This means that you don't need to restart the app every time you make a change to the Python code. You can see how it's happening in real-time in the Next.js terminal.

## Parting Words

Keep in mind that you fully own all of this code! This entire devtool system runs locally, so feel free to modify it however you want. This is just designed as scaffolding to help you get started.

## Quick Links

- **📚 Documentation**: See [`docs/README.md`](docs/README.md) for all guides
- **🚀 Quick Start**: See [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md)
- **🎯 Training**: See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)
- **📊 Benchmarking**: See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md)

## Common Commands

```bash
# Train model on GPU (recommended)
modal run training/modal_train.py

# Benchmark bot strength
python benchmarks/benchmark_stockfish.py --progressive

# Run devtools
cd devtools && npm run dev
```

## Getting Help

If you need further help:
1. Check the [documentation](docs/README.md) in `docs/`
2. See [ChessHacks docs](https://docs.chesshacks.dev/)
3. Join [ChessHacks Discord](https://docs.chesshacks.dev/resources/discord)
