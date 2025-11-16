#!/usr/bin/env python3
"""
Wrapper script to run training pipeline from root directory.
Usage: python run_training.py --gpu
"""

import sys
from pathlib import Path

# Add training directory to path
training_dir = Path(__file__).parent / 'training'
sys.path.insert(0, str(training_dir))

# Import and run pipeline
from pipeline import run_full_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess bot training pipeline")

    # General options
    parser.add_argument('--phase', type=str, choices=['all', '2', '3'],
                       default='all', help='Which phase(s) to run')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')

    # Phase 2 options
    parser.add_argument('--selfplay-games', type=int, default=1000,
                       help='Number of self-play games (Phase 2)')
    parser.add_argument('--selfplay-sims', type=int, default=100,
                       help='MCTS simulations per move (Phase 2)')
    parser.add_argument('--selfplay-epochs', type=int, default=100,
                       help='Training epochs for Phase 2')

    # Phase 3 options
    parser.add_argument('--tactical-puzzles', type=int, default=50000,
                       help='Number of tactical puzzles (Phase 3)')
    parser.add_argument('--tactical-epochs', type=int, default=50,
                       help='Training epochs for Phase 3')

    args = parser.parse_args()

    skip_phase2 = args.phase == '3'
    skip_phase3 = args.phase == '2'

    run_full_pipeline(
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        selfplay_epochs=args.selfplay_epochs,
        tactical_puzzles=args.tactical_puzzles,
        tactical_epochs=args.tactical_epochs,
        skip_phase2=skip_phase2,
        skip_phase3=skip_phase3,
        use_gpu=args.gpu
    )
