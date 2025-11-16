"""
Phase 2 Model Comparison Script
Compare two chess models on:
1. Policy accuracy
2. Value MAE (Mean Absolute Error)
3. Head-to-head gameplay
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from chess import Board
import sys
import os
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.model import ChessModel
from src.utils.board_encoder import board_to_tensor_torch
from src.utils.move_mapper import MoveMapper
from src.utils.mcts import MCTS


def load_model(model_path, device='cpu'):
    """Load a chess model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ChessModel(
        num_residual_blocks=checkpoint.get('num_residual_blocks', 6),
        channels=checkpoint.get('channels', 64),
        dropout=0.0  # No dropout for evaluation
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_policy_accuracy(model1, model2, num_positions=100):
    """
    Compare policy accuracy: how often do models agree on best move?
    """
    print(f"\n{'='*70}")
    print("POLICY ACCURACY COMPARISON")
    print(f"{'='*70}")

    move_mapper = MoveMapper()
    agreements = 0

    for i in range(num_positions):
        # Create random position
        board = Board()
        for _ in range(random.randint(0, 20)):
            legal_moves = list(board.legal_moves)
            if legal_moves and not board.is_game_over():
                board.push(random.choice(legal_moves))

        if board.is_game_over():
            continue

        # Get predictions from both models
        board_tensor = board_to_tensor_torch(board)
        legal_moves = list(board.legal_moves)

        with torch.no_grad():
            policy1, _ = model1(board_tensor)
            policy2, _ = model2(board_tensor)

        # Get best moves
        probs1 = move_mapper.get_move_probabilities(policy1[0].cpu().numpy(), legal_moves)
        probs2 = move_mapper.get_move_probabilities(policy2[0].cpu().numpy(), legal_moves)

        best_move1 = max(probs1.items(), key=lambda x: x[1])[0]
        best_move2 = max(probs2.items(), key=lambda x: x[1])[0]

        if best_move1 == best_move2:
            agreements += 1

    accuracy = agreements / num_positions * 100
    print(f"  Policy Agreement: {accuracy:.1f}% ({agreements}/{num_positions} positions)")
    print(f"  → Higher agreement suggests similar playing style")
    print(f"  → Lower agreement suggests model2 learned new strategies")

    return accuracy


def evaluate_value_mae(model1, model2, num_positions=100):
    """
    Compare value prediction accuracy (Mean Absolute Error).
    """
    print(f"\n{'='*70}")
    print("VALUE PREDICTION COMPARISON")
    print(f"{'='*70}")

    value_diffs = []

    for i in range(num_positions):
        # Create random position
        board = Board()
        for _ in range(random.randint(0, 30)):
            legal_moves = list(board.legal_moves)
            if legal_moves and not board.is_game_over():
                board.push(random.choice(legal_moves))

        if board.is_game_over():
            continue

        # Get value predictions
        board_tensor = board_to_tensor_torch(board)

        with torch.no_grad():
            _, value1 = model1(board_tensor)
            _, value2 = model2(board_tensor)

        diff = abs(value1.item() - value2.item())
        value_diffs.append(diff)

    mae = np.mean(value_diffs)
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  → Lower MAE suggests similar position evaluation")
    print(f"  → Higher MAE suggests model2 evaluates positions differently")

    return mae


def play_game(model1, model2, mcts_sims=20):
    """
    Play one game between two models.
    Returns: 1 if model1 wins, -1 if model2 wins, 0 if draw
    """
    board = Board()
    move_mapper = MoveMapper()
    mcts1 = MCTS(model1, move_mapper, num_simulations=mcts_sims)
    mcts2 = MCTS(model2, move_mapper, num_simulations=mcts_sims)

    max_moves = 100
    moves = 0

    while not board.is_game_over() and moves < max_moves:
        # White's turn (model1)
        if board.turn:
            best_move, _ = mcts1.search(board)
        # Black's turn (model2)
        else:
            best_move, _ = mcts2.search(board)

        board.push(best_move)
        moves += 1

    # Determine result
    if board.is_checkmate():
        return 1 if not board.turn else -1  # Winner is opposite of current turn
    else:
        return 0  # Draw


def evaluate_head_to_head(model1, model2, num_games=10, mcts_sims=20):
    """
    Play head-to-head games between models.
    """
    print(f"\n{'='*70}")
    print(f"HEAD-TO-HEAD GAMEPLAY ({num_games} games)")
    print(f"{'='*70}")
    print(f"  MCTS simulations per move: {mcts_sims}")
    print(f"  Model1 plays White, Model2 plays Black\n")

    results = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}

    for game_num in range(num_games):
        print(f"  Game {game_num + 1}/{num_games}...", end=' ', flush=True)
        result = play_game(model1, model2, mcts_sims)

        if result == 1:
            results['model1_wins'] += 1
            print("Model1 wins")
        elif result == -1:
            results['model2_wins'] += 1
            print("Model2 wins")
        else:
            results['draws'] += 1
            print("Draw")

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Model1 Wins: {results['model1_wins']}")
    print(f"  Model2 Wins: {results['model2_wins']}")
    print(f"  Draws: {results['draws']}")

    win_rate = results['model2_wins'] / num_games * 100
    print(f"\n  Model2 Win Rate: {win_rate:.1f}%")

    if win_rate > 55:
        print(f"  ✅ Model2 is STRONGER - consider using it as new baseline!")
    elif win_rate < 45:
        print(f"  ⚠️  Model2 is WEAKER - keep training or use more data")
    else:
        print(f"  ➡️  Models are SIMILAR - may need more games to decide")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare two chess models')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model (baseline)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model (new model)')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of head-to-head games (default: 10)')
    parser.add_argument('--mcts-sims', type=int, default=20,
                       help='MCTS simulations per move (default: 20)')
    parser.add_argument('--skip-policy', action='store_true',
                       help='Skip policy accuracy test')
    parser.add_argument('--skip-value', action='store_true',
                       help='Skip value MAE test')
    parser.add_argument('--skip-games', action='store_true',
                       help='Skip head-to-head games')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🎯 PHASE 2: MODEL COMPARISON")
    print("="*70)
    print(f"Model 1 (Baseline): {args.model1}")
    print(f"Model 2 (New):      {args.model2}")
    print("="*70)

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading models on {device}...")

    model1 = load_model(args.model1, device)
    model2 = load_model(args.model2, device)

    print(f"  ✓ Model1: {count_parameters(model1):,} parameters")
    print(f"  ✓ Model2: {count_parameters(model2):,} parameters")

    # Run evaluations
    if not args.skip_policy:
        policy_acc = evaluate_policy_accuracy(model1, model2, num_positions=100)

    if not args.skip_value:
        value_mae = evaluate_value_mae(model1, model2, num_positions=100)

    if not args.skip_games:
        results = evaluate_head_to_head(model1, model2, args.games, args.mcts_sims)

    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

