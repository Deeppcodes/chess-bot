"""
Complete training pipeline orchestrator.
Runs all phases sequentially: Phase 1 (Lichess) → Phase 2 (Self-Play) → Phase 3 (Tactics)
"""

import argparse
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def phase1_lichess_training():
    """
    Phase 1: Supervised learning on Lichess games.
    Already completed - using chess_model_best.pth
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Supervised Learning on Lichess Games")
    print("=" * 70)
    print("Status: ✓ Already completed")
    print("Model: chess_model_best.pth")
    print("This phase trained on 100K+ Lichess games (2000+ rating)")
    print("=" * 70)


def phase2_selfplay(
    num_games: int = 1000,
    num_simulations: int = 100,
    epochs: int = 100,
    use_gpu: bool = False
):
    """
    Phase 2: Self-play reinforcement learning.

    Args:
        num_games: Number of self-play games to generate
        num_simulations: MCTS simulations per move
        epochs: Training epochs
        use_gpu: Whether to use GPU
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Self-Play Reinforcement Learning")
    print("=" * 70)

    # Step 1: Generate self-play data
    print("\nStep 1: Generating self-play games...")
    print(f"  Games: {num_games}")
    print(f"  MCTS simulations: {num_simulations}")

    # Import from current directory
    import selfplay_generator
    from src.utils.improved_model import ImprovedChessModel
    from src.utils.move_mapper import MoveMapper
    import torch

    # Load Phase 1 model
    model_path = project_root / 'chess_model_best.pth'
    checkpoint = torch.load(str(model_path), map_location='cpu')
    model = ImprovedChessModel(hidden_size=checkpoint.get('hidden_size', 512))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    move_mapper = MoveMapper()

    # Generate self-play data
    generator = selfplay_generator.SelfPlayDataGenerator(model, move_mapper, num_simulations=num_simulations)
    data_path = project_root / 'data' / 'selfplay.npz'
    generator.generate_games(
        num_games=num_games,
        output_path=str(data_path),
        verbose=True
    )

    # Step 2: Train on self-play data
    print("\nStep 2: Training on self-play data...")

    import train_selfplay

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    train_selfplay.train_on_selfplay(
        model_path=str(model_path),
        selfplay_data_path=str(data_path),
        output_path=str(project_root / 'models' / 'chess_model_phase2.pth'),
        epochs=epochs,
        batch_size=256,
        lr=0.0001,
        device=device
    )

    print("\n✅ Phase 2 complete!")
    print("=" * 70)


def phase3_tactical(
    num_puzzles: int = 50000,
    epochs: int = 50,
    use_gpu: bool = False
):
    """
    Phase 3: Tactical puzzle training.

    Args:
        num_puzzles: Number of tactical puzzles to download
        epochs: Training epochs
        use_gpu: Whether to use GPU
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Tactical Puzzle Training")
    print("=" * 70)

    # Step 1: Download tactical puzzles
    print("\nStep 1: Downloading tactical puzzles...")
    print(f"  Target puzzles: {num_puzzles}")

    import tactical_puzzles

    tactical_data_path = project_root / 'data' / 'tactical_puzzles.npz'
    tactical_puzzles.download_lichess_puzzles(
        num_puzzles=num_puzzles,
        min_rating=1500,
        max_rating=2500,
        output_path=str(tactical_data_path),
        verbose=True
    )

    # Step 2: Train on tactical puzzles
    print("\nStep 2: Training on tactical puzzles...")

    import train_tactical
    import torch

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    # Use Phase 2 model if available, otherwise Phase 1
    phase2_path = project_root / 'models' / 'chess_model_phase2.pth'
    phase1_path = project_root / 'chess_model_best.pth'

    if phase2_path.exists():
        input_model = str(phase2_path)
        print(f"  Using Phase 2 model as input")
    else:
        input_model = str(phase1_path)
        print(f"  Using Phase 1 model as input (Phase 2 skipped)")

    train_tactical.train_on_tactics(
        model_path=input_model,
        tactical_data_path=str(tactical_data_path),
        output_path=str(project_root / 'models' / 'chess_model_phase3.pth'),
        epochs=epochs,
        batch_size=512,
        lr=0.0001,
        policy_weight=2.0,
        device=device
    )

    print("\n✅ Phase 3 complete!")
    print("=" * 70)


def run_full_pipeline(
    selfplay_games: int = 1000,
    selfplay_sims: int = 100,
    selfplay_epochs: int = 100,
    tactical_puzzles: int = 50000,
    tactical_epochs: int = 50,
    skip_phase2: bool = False,
    skip_phase3: bool = False,
    use_gpu: bool = False
):
    """
    Run the complete training pipeline.

    Args:
        selfplay_games: Number of self-play games (Phase 2)
        selfplay_sims: MCTS simulations per move (Phase 2)
        selfplay_epochs: Training epochs for Phase 2
        tactical_puzzles: Number of tactical puzzles (Phase 3)
        tactical_epochs: Training epochs for Phase 3
        skip_phase2: Skip Phase 2 (self-play)
        skip_phase3: Skip Phase 3 (tactics)
        use_gpu: Use GPU if available
    """
    # Create necessary directories
    (project_root / 'data').mkdir(exist_ok=True)
    (project_root / 'models').mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print(" " * 20 + "CHESS BOT TRAINING PIPELINE")
    print("=" * 80)

    # Phase 1
    phase1_lichess_training()

    # Phase 2
    if not skip_phase2:
        phase2_selfplay(
            num_games=selfplay_games,
            num_simulations=selfplay_sims,
            epochs=selfplay_epochs,
            use_gpu=use_gpu
        )
    else:
        print("\n⏭️  Skipping Phase 2 (self-play)")

    # Phase 3
    if not skip_phase3:
        phase3_tactical(
            num_puzzles=tactical_puzzles,
            epochs=tactical_epochs,
            use_gpu=use_gpu
        )
    else:
        print("\n⏭️  Skipping Phase 3 (tactical)")

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)

    final_model = None
    phase3_path = project_root / 'models' / 'chess_model_phase3.pth'
    phase2_path = project_root / 'models' / 'chess_model_phase2.pth'
    phase1_path = project_root / 'chess_model_best.pth'

    if not skip_phase3 and phase3_path.exists():
        final_model = str(phase3_path)
    elif not skip_phase2 and phase2_path.exists():
        final_model = str(phase2_path)
    else:
        final_model = str(phase1_path)

    print(f"\nFinal model: {final_model}")
    print("\nTo use the new model:")
    print(f"  1. Copy to main directory: cp {final_model} chess_model_best.pth")
    print(f"  2. Test your bot: python src/main.py")
    print("\nExpected improvements:")
    if not skip_phase2:
        print("  ✅ Phase 2 (Self-Play): +50-100 ELO")
    if not skip_phase3:
        print("  ✅ Phase 3 (Tactical): +100-150 ELO")
    if not skip_phase2 and not skip_phase3:
        print("  📊 Total expected gain: +150-250 ELO")

    print("\n" + "=" * 80)


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
