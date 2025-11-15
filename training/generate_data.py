"""
Script to generate training data from random self-play games.
Run this to create training_data.npz file.
"""

import argparse
from src.utils.data_collection import generate_training_data, save_training_data


def main():
    parser = argparse.ArgumentParser(description='Generate training data from random chess games')
    parser.add_argument('--num-games', type=int, default=100, 
                       help='Number of games to generate (default: 100)')
    parser.add_argument('--max-moves', type=int, default=200,
                       help='Maximum moves per game (default: 200)')
    parser.add_argument('--output', type=str, default='training_data.npz',
                       help='Output filename (default: training_data.npz)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 2: Training Data Generation")
    print("=" * 60)
    print(f"Generating {args.num_games} games with max {args.max_moves} moves each...")
    print()
    
    # Generate training data
    board_states, moves, outcomes = generate_training_data(
        num_games=args.num_games,
        max_moves=args.max_moves
    )
    
    # Save to file
    save_training_data(board_states, moves, outcomes, args.output)
    
    print()
    print("=" * 60)
    print(f"✓ Training data generation complete!")
    print(f"  Output file: {args.output}")
    print(f"  Total positions: {len(board_states)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

