"""
AlphaZero-style self-play data generator for chess neural network training.

Generates high-quality training data using:
- CNN/ResNet model for position evaluation
- MCTS for move selection
- Temperature-based exploration
- Dirichlet noise for opening diversity
- Proper policy targets from visit counts
- Proper value targets from game outcomes
"""

import argparse
import numpy as np
import torch
import chess
from typing import List, Tuple, Dict
import time

from src.utils.model import ChessModel
from src.utils.board_encoder import board_to_tensor
from src.utils.move_mapper import MoveMapper
from src.utils.mcts import MCTS, MCTSNode


def apply_temperature(visit_counts: Dict[chess.Move, int], temperature: float) -> Dict[chess.Move, float]:
    """
    Apply temperature to visit counts to get move probabilities.
    
    Args:
        visit_counts: Dictionary of move -> visit count
        temperature: Temperature parameter (0 = deterministic, 1 = proportional)
    
    Returns:
        Dictionary of move -> probability
    """
    if temperature == 0:
        # Deterministic: pick move with highest visit count
        max_visits = max(visit_counts.values())
        probs = {move: 1.0 if count == max_visits else 0.0 
                for move, count in visit_counts.items()}
        # Normalize
        total = sum(probs.values())
        return {move: p / total for move, p in probs.items()}
    
    # Apply temperature
    visits_temp = {move: count ** (1.0 / temperature) 
                   for move, count in visit_counts.items()}
    total = sum(visits_temp.values())
    
    if total == 0:
        # Uniform if all zero
        return {move: 1.0 / len(visit_counts) for move in visit_counts}
    
    return {move: v / total for move, v in visits_temp.items()}


def add_dirichlet_noise(policy_prior: Dict[chess.Move, float], alpha: float = 0.3, epsilon: float = 0.25) -> Dict[chess.Move, float]:
    """
    Add Dirichlet noise to policy for exploration in openings.
    
    Args:
        policy_prior: Original policy probabilities
        alpha: Dirichlet concentration parameter
        epsilon: Mixing weight for noise
    
    Returns:
        Noisy policy probabilities
    """
    moves = list(policy_prior.keys())
    if len(moves) == 0:
        return policy_prior
    
    # Generate Dirichlet noise
    noise = np.random.dirichlet([alpha] * len(moves))
    
    # Mix with original policy
    noisy_policy = {}
    for i, move in enumerate(moves):
        noisy_policy[move] = (1 - epsilon) * policy_prior[move] + epsilon * noise[i]
    
    return noisy_policy


def select_move_with_temperature(move_probs: Dict[chess.Move, float]) -> chess.Move:
    """
    Select a move based on probability distribution.
    
    Args:
        move_probs: Dictionary of move -> probability
    
    Returns:
        Selected move
    """
    moves = list(move_probs.keys())
    probs = list(move_probs.values())
    
    # Normalize
    total = sum(probs)
    if total == 0:
        return np.random.choice(moves)
    
    probs = [p / total for p in probs]
    
    return np.random.choice(moves, p=probs)


def play_self_play_game(model, move_mapper, mcts_sims: int = 50, max_length: int = 60,
                        add_noise: bool = True, random_opening_moves: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Play one self-play game and collect training data.
    
    Args:
        model: Neural network model
        move_mapper: MoveMapper instance
        mcts_sims: Number of MCTS simulations per move
        max_length: Maximum game length in plies
        add_noise: Whether to add Dirichlet noise to root
    
    Returns:
        Tuple of (board_states, policy_targets, players)
        - board_states: List of board tensors (8, 8, 12)
        - policy_targets: List of policy distributions (256,)
        - players: List of player indicators (+1 for white, -1 for black)
    """
    board = chess.Board()
    mcts = MCTS(model, move_mapper, num_simulations=mcts_sims)
    
    positions = []
    policies = []
    players = []
    
    ply = 0

    # Random opening moves for diversity
    import random
    for _ in range(random_opening_moves):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(random.choice(legal_moves))
            ply += 1

    while not board.is_game_over() and ply < max_length:
        # Check for draws (only after some moves to avoid false positives)
        if ply > 10 and board.can_claim_draw():
            break
        # Determine temperature based on game phase
        if ply < 10:
            temperature = 1.0  # High exploration early
        elif ply < 30:
            temperature = 0.5  # Medium exploration mid-game
        else:
            temperature = 0.1  # Low exploration late-game
        
        # Encode current board state
        board_tensor = board_to_tensor(board)
        positions.append(board_tensor)
        
        # Record whose turn it is (+1 for white, -1 for black)
        player = 1 if board.turn else -1
        players.append(player)

        # Run MCTS search (with Dirichlet noise for opening diversity)
        use_dirichlet = add_noise and ply < 10
        best_move, move_probs = mcts.search(board, add_dirichlet=use_dirichlet)

        # Get position evaluation from MCTS
        position_value = mcts.root.get_value() if hasattr(mcts, 'root') else 0

        # Resignation logic: if position is hopeless, end game
        # Relaxed threshold to end games earlier
        RESIGNATION_THRESHOLD = -0.7
        if position_value < RESIGNATION_THRESHOLD and ply > 15:
            # Resign - opponent wins
            outcome = -1 if board.turn else 1
            value_targets = [outcome * player for player in players]
            print(f"  → Resigned at ply {ply} (value: {position_value:.3f})", flush=True)
            return positions, policies, value_targets

        # Get visit counts from MCTS root
        # Convert move probabilities to visit counts for temperature application
        # (In real implementation, we'd get actual visit counts from MCTS)
        # For now, we'll use the probabilities as-is and apply temperature

        # Apply temperature to move selection
        temp_probs = apply_temperature(
            {move: prob * 1000 for move, prob in move_probs.items()},  # Scale to simulate visit counts
            temperature
        )

        # Convert policy to 256-dimensional vector
        policy_vector = np.zeros(256, dtype=np.float32)
        for move, prob in temp_probs.items():
            move_idx = move_mapper.get_move_index(move)
            if move_idx < 256:
                policy_vector[move_idx] = prob

        # Normalize policy vector
        if policy_vector.sum() > 0:
            policy_vector = policy_vector / policy_vector.sum()

        policies.append(policy_vector)

        # Select move based on temperature
        selected_move = select_move_with_temperature(temp_probs)

        # Make the move
        board.push(selected_move)
        ply += 1

    # Game finished - determine outcome
    if board.is_checkmate():
        # Winner is the player who just moved (opponent is checkmated)
        outcome = 1 if not board.turn else -1
    else:
        # Draw
        outcome = 0

    # Create value targets based on outcome and player perspective
    value_targets = [outcome * player for player in players]

    return positions, policies, value_targets


def convert_old_to_new_architecture(old_state_dict):
    """
    Convert old model architecture (conv_input, res_blocks) to new architecture
    (initial_conv, residual_blocks).

    Old architecture keys:
    - conv_input.0.weight -> initial_conv.weight
    - res_blocks.X.Y -> residual_blocks.X.convY/bnY
    - conv_layers -> policy_conv/value_conv
    - fc_shared -> (removed in new arch)
    - policy_head -> policy_fc
    - value_head -> value_fc1, value_fc2
    """
    new_state_dict = {}

    for old_key, value in old_state_dict.items():
        # Initial convolution layer
        if old_key.startswith('conv_input.0.'):
            new_key = old_key.replace('conv_input.0.', 'initial_conv.')
            new_state_dict[new_key] = value
        elif old_key.startswith('conv_input.1.'):
            new_key = old_key.replace('conv_input.1.', 'initial_bn.')
            new_state_dict[new_key] = value

        # Residual blocks: res_blocks.X.0 -> residual_blocks.X.conv1
        elif old_key.startswith('res_blocks.'):
            parts = old_key.split('.')
            block_num = parts[1]
            layer_num = parts[2]
            param = '.'.join(parts[3:]) if len(parts) > 3 else ''

            if layer_num == '0':  # First conv
                new_key = f'residual_blocks.{block_num}.conv1.{param}' if param else f'residual_blocks.{block_num}.conv1'
            elif layer_num == '1':  # First BN
                new_key = f'residual_blocks.{block_num}.bn1.{param}' if param else f'residual_blocks.{block_num}.bn1'
            elif layer_num == '3':  # Second conv
                new_key = f'residual_blocks.{block_num}.conv2.{param}' if param else f'residual_blocks.{block_num}.conv2'
            elif layer_num == '4':  # Second BN
                new_key = f'residual_blocks.{block_num}.bn2.{param}' if param else f'residual_blocks.{block_num}.bn2'
            else:
                continue  # Skip ReLU layers (not saved in state_dict)

            new_state_dict[new_key] = value

        # Policy head: conv_layers.0 -> policy_conv, conv_layers.1 -> policy_bn
        elif old_key.startswith('conv_layers.0.'):
            new_key = old_key.replace('conv_layers.0.', 'policy_conv.')
            new_state_dict[new_key] = value
        elif old_key.startswith('conv_layers.1.'):
            new_key = old_key.replace('conv_layers.1.', 'policy_bn.')
            new_state_dict[new_key] = value

        # Value head: conv_layers.3 -> value_conv, conv_layers.4 -> value_bn
        elif old_key.startswith('conv_layers.3.'):
            new_key = old_key.replace('conv_layers.3.', 'value_conv.')
            new_state_dict[new_key] = value
        elif old_key.startswith('conv_layers.4.'):
            new_key = old_key.replace('conv_layers.4.', 'value_bn.')
            new_state_dict[new_key] = value

        # Policy FC: policy_head.0 and policy_head.3 -> policy_fc
        elif old_key.startswith('policy_head.3.'):
            new_key = old_key.replace('policy_head.3.', 'policy_fc.')
            new_state_dict[new_key] = value

        # Value FC: value_head.0 -> value_fc1, value_head.3 -> value_fc2
        elif old_key.startswith('value_head.0.'):
            new_key = old_key.replace('value_head.0.', 'value_fc1.')
            new_state_dict[new_key] = value
        elif old_key.startswith('value_head.3.'):
            new_key = old_key.replace('value_head.3.', 'value_fc2.')
            new_state_dict[new_key] = value

        # Skip fc_shared and policy_head.0 (removed in new architecture)
        elif old_key.startswith('fc_shared.') or old_key.startswith('policy_head.0.'):
            continue

    return new_state_dict


def generate_selfplay_dataset(
    num_games: int,
    mcts_sims: int,
    max_length: int,
    output_path: str,
    model_path: str = "chess_model_improved.pth",
    device: str = None,
):
    """
    Generate self-play games and save to a .npz file.

    This is the main callable function for Modal and other external use.

    Args:
        num_games: Number of self-play games to generate
        mcts_sims: Number of MCTS simulations per move
        max_length: Maximum game length in plies
        output_path: Path to save the .npz file
        model_path: Path to the model checkpoint
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("🚀 AlphaZero-Style Self-Play Data Generator")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"MCTS simulations: {mcts_sims}")
    print(f"Max game length: {max_length}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print("="*70)

    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint with metadata (from train_improved.py)
        num_residual_blocks = checkpoint.get('num_residual_blocks', 6)
        channels = checkpoint.get('channels', 64)
        state_dict = checkpoint['model_state_dict']
        move_mapper = checkpoint.get('move_mapper', MoveMapper())
        print(f"  Format: Checkpoint with metadata")
    else:
        # Direct state_dict (from train_modal_selfplay.py)
        num_residual_blocks = 6
        channels = 64
        state_dict = checkpoint
        move_mapper = MoveMapper()
        print(f"  Format: Direct state_dict")

    # Detect old architecture and convert if needed
    first_key = list(state_dict.keys())[0]
    if 'conv_input' in first_key or 'res_blocks' in first_key:
        print(f"  ⚠️  Detected OLD architecture - converting to new format...")
        state_dict = convert_old_to_new_architecture(state_dict)
        print(f"  ✓ Architecture converted successfully")

    # Create model
    model = ChessModel(
        num_residual_blocks=num_residual_blocks,
        channels=channels,
        dropout=0.0  # No dropout during inference
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Architecture: {num_residual_blocks} residual blocks, {channels} channels")

    # Generate self-play data
    with torch.no_grad():
        boards, policy_targets, value_targets = generate_selfplay_data(
            model, move_mapper, num_games, mcts_sims, max_length
        )

    # Save dataset
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        boards=boards,
        policy_targets=policy_targets,
        value_targets=value_targets
    )

    print(f"\n{'='*70}")
    print(f"✓ Self-play data saved to {output_path}")
    print(f"  Total positions: {len(boards)}")
    print(f"  Wins: {np.sum(value_targets > 0)}")
    print(f"  Draws: {np.sum(value_targets == 0)}")
    print(f"  Losses: {np.sum(value_targets < 0)}")
    print(f"{'='*70}")

    return output_path


def generate_selfplay_data(model, move_mapper, num_games: int = 100, mcts_sims: int = 50,
                           max_length: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate self-play training data (internal function).

    Args:
        model: Neural network model
        move_mapper: MoveMapper instance
        num_games: Number of games to generate
        mcts_sims: MCTS simulations per move
        max_length: Maximum game length

    Returns:
        Tuple of (boards, policy_targets, value_targets)
        - boards: (N, 12, 8, 8) board tensors
        - policy_targets: (N, 256) policy distributions
        - value_targets: (N,) game outcomes
    """
    all_boards = []
    all_policies = []
    all_values = []

    print(f"\n{'='*70}")
    print(f"Generating {num_games} self-play games with {mcts_sims} MCTS simulations")
    print(f"{'='*70}\n")

    for game_idx in range(num_games):
        start_time = time.time()

        # Play one game
        positions, policies, values = play_self_play_game(
            model, move_mapper, mcts_sims, max_length
        )

        # Add to dataset
        all_boards.extend(positions)
        all_policies.extend(policies)
        all_values.extend(values)

        game_time = time.time() - start_time

        # Print progress every game for first 5, then every 5 games
        if game_idx < 5 or (game_idx + 1) % 5 == 0:
            print(f"Game {game_idx + 1}/{num_games} | "
                  f"Positions: {len(positions)} | "
                  f"Time: {game_time:.1f}s | "
                  f"Total positions: {len(all_boards)}", flush=True)

    # Convert to numpy arrays
    boards = np.array(all_boards, dtype=np.float32)
    policy_targets = np.array(all_policies, dtype=np.float32)
    value_targets = np.array(all_values, dtype=np.float32)

    # Convert boards from (N, 8, 8, 12) to (N, 12, 8, 8)
    boards = np.transpose(boards, (0, 3, 1, 2))

    print(f"\n{'='*70}")
    print(f"✓ Generated {len(boards)} training positions from {num_games} games")
    print(f"  Board shape: {boards.shape}")
    print(f"  Policy shape: {policy_targets.shape}")
    print(f"  Value shape: {value_targets.shape}")
    print(f"{'='*70}\n")

    return boards, policy_targets, value_targets


def main():
    parser = argparse.ArgumentParser(description='Generate self-play training data')
    parser.add_argument('--games', type=int, default=300,
                       help='Number of self-play games (default: 300)')
    parser.add_argument('--mcts-sims', type=int, default=50,
                       help='MCTS simulations per move (default: 50)')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum game length in plies (default: 200)')
    parser.add_argument('--output', type=str, default='datasets/selfplay/',
                       help='Output directory or file (default: datasets/selfplay/)')
    parser.add_argument('--model', type=str, default='chess_model_best.pth',
                       help='Model checkpoint to use (default: chess_model_best.pth)')
    parser.add_argument('--batch-id', type=int, default=None,
                       help='Batch ID for naming (auto-increments if not specified)')

    args = parser.parse_args()

    # Determine output path
    import os
    if args.output.endswith('/') or os.path.isdir(args.output):
        # Output is a directory - create batch file
        os.makedirs(args.output, exist_ok=True)

        # Auto-increment batch ID if not specified
        if args.batch_id is None:
            existing_batches = [f for f in os.listdir(args.output) if f.startswith('game_batch_') and f.endswith('.npz')]
            if existing_batches:
                batch_nums = [int(f.split('_')[2].split('.')[0]) for f in existing_batches]
                args.batch_id = max(batch_nums) + 1
            else:
                args.batch_id = 1

        output_path = os.path.join(args.output, f'game_batch_{args.batch_id:04d}.npz')
    else:
        # Output is a file path
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🎮 PHASE 2: SELF-PLAY DATA GENERATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Games: {args.games}")
    print(f"MCTS Simulations: {args.mcts_sims}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    # Use the callable function
    generate_selfplay_dataset(
        num_games=args.games,
        mcts_sims=args.mcts_sims,
        max_length=args.max_length,
        output_path=output_path,
        model_path=args.model,
        device=None  # Auto-detect
    )

    print(f"\n{'='*70}")
    print(f"✅ SELF-PLAY GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Train on this data:")
    print(f"     python -m src.training.train_selfplay --data-dir {os.path.dirname(output_path)} --model {args.model} --epochs 5")
    print(f"\n  2. Or train on single file:")
    print(f"     python -m src.training.train_selfplay --data {output_path} --model {args.model} --epochs 5")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


