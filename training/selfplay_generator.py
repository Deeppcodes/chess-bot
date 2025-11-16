"""
Self-play data generator for Phase 2 training.
Uses MCTS to play games against itself and generate training data.
"""

import chess
import numpy as np
import torch
from typing import List, Tuple, Dict
import random
from pathlib import Path
import time


class SelfPlayGame:
    """Generates a single self-play game."""

    def __init__(self, model, move_mapper, mcts, temperature_schedule=None):
        """
        Initialize self-play game generator.

        Args:
            model: Neural network model
            move_mapper: MoveMapper instance
            mcts: MCTS search instance
            temperature_schedule: Function move_number -> temperature (default: step function)
        """
        self.model = model
        self.move_mapper = move_mapper
        self.mcts = mcts

        # Default temperature schedule: high for first 30 moves, then low
        if temperature_schedule is None:
            self.temperature_schedule = lambda move_num: 1.0 if move_num < 30 else 0.1
        else:
            self.temperature_schedule = temperature_schedule

    def play_game(self, max_moves: int = 200, verbose: bool = False) -> Dict:
        """
        Play a complete self-play game.

        Args:
            max_moves: Maximum number of moves before declaring draw
            verbose: Print game progress

        Returns:
            Dictionary containing:
                - positions: List of board states
                - policies: List of MCTS policy vectors
                - game_result: Final result (1.0, -1.0, or 0.0)
                - move_count: Number of moves played
        """
        board = chess.Board()

        positions = []
        policies = []
        moves_played = []

        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            # Get current position encoding
            from src.utils.board_encoder import board_to_tensor
            position = board_to_tensor(board)

            # Run MCTS search
            best_move, move_probs = self.mcts.search(board)

            # Apply temperature
            temperature = self.temperature_schedule(move_count)
            if temperature != 1.0:
                move_probs = self._apply_temperature(move_probs, temperature)

            # Convert move probabilities to policy vector
            policy_vector = self._move_probs_to_policy_vector(board, move_probs)

            # Store position and policy
            positions.append(position)
            policies.append(policy_vector)

            # Sample move from policy (with temperature)
            if temperature > 0.5:
                # Stochastic: sample from distribution
                move = self._sample_move(move_probs)
            else:
                # Deterministic: always best move
                move = best_move

            # Make move
            moves_played.append(move.uci())
            board.push(move)
            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"  Move {move_count}: {move.uci()}")

        # Determine game result
        if board.is_checkmate():
            # Winner is the player who just moved (opponent is checkmated)
            winner = not board.turn  # Opponent's color
            game_result = 1.0 if winner == chess.WHITE else -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or move_count >= max_moves:
            game_result = 0.0  # Draw
        elif board.can_claim_draw():
            game_result = 0.0  # Draw
        else:
            game_result = 0.0  # Draw by default

        if verbose:
            result_str = {1.0: "White wins", -1.0: "Black wins", 0.0: "Draw"}[game_result]
            print(f"Game finished: {result_str} in {move_count} moves")

        return {
            'positions': positions,
            'policies': policies,
            'game_result': game_result,
            'move_count': move_count,
            'moves': moves_played,
            'final_board': board.fen()
        }

    def _apply_temperature(self, move_probs: Dict[chess.Move, float],
                          temperature: float) -> Dict[chess.Move, float]:
        """Apply temperature to move probabilities."""
        if temperature == 0:
            # Deterministic: max probability to best move
            best_move = max(move_probs.items(), key=lambda x: x[1])[0]
            return {move: (1.0 if move == best_move else 0.0) for move in move_probs}

        # Apply temperature
        adjusted_probs = {}
        for move, prob in move_probs.items():
            adjusted_probs[move] = prob ** (1.0 / temperature)

        # Renormalize
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {move: prob / total for move, prob in adjusted_probs.items()}

        return adjusted_probs

    def _sample_move(self, move_probs: Dict[chess.Move, float]) -> chess.Move:
        """Sample a move from probability distribution."""
        moves = list(move_probs.keys())
        probs = [move_probs[move] for move in moves]

        # Normalize to ensure sum = 1.0
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(moves)] * len(moves)

        return np.random.choice(moves, p=probs)

    def _move_probs_to_policy_vector(self, board: chess.Board,
                                     move_probs: Dict[chess.Move, float]) -> np.ndarray:
        """
        Convert move probabilities to policy vector (4096 dimensions).
        Uses unified encoding: from_square * 64 + to_square.
        """
        policy = np.zeros(4096, dtype=np.float32)

        for move, prob in move_probs.items():
            idx = move.from_square * 64 + move.to_square
            policy[idx] = prob

        return policy


class SelfPlayDataGenerator:
    """Generates multiple self-play games for training."""

    def __init__(self, model, move_mapper, num_simulations: int = 100):
        """
        Initialize self-play data generator.

        Args:
            model: Neural network model
            move_mapper: MoveMapper instance
            num_simulations: MCTS simulations per move
        """
        self.model = model
        self.move_mapper = move_mapper
        self.num_simulations = num_simulations

    def generate_games(self, num_games: int, output_path: str = None,
                      verbose: bool = True) -> Dict:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            output_path: Path to save data (optional)
            verbose: Print progress

        Returns:
            Dictionary with training data
        """
        from src.utils.mcts import MCTS

        # Create MCTS instance
        mcts = MCTS(self.model, self.move_mapper,
                   num_simulations=self.num_simulations,
                   exploration_constant=1.5)

        all_positions = []
        all_policies = []
        all_values = []

        if verbose:
            print(f"Generating {num_games} self-play games...")
            print(f"MCTS simulations per move: {self.num_simulations}")

        start_time = time.time()

        for game_idx in range(num_games):
            game_gen = SelfPlayGame(self.model, self.move_mapper, mcts)
            game_data = game_gen.play_game(verbose=False)

            # Extract data
            positions = game_data['positions']
            policies = game_data['policies']
            game_result = game_data['game_result']

            # Assign values based on game result
            # Value for each position depends on whose turn it was
            values = []
            for i in range(len(positions)):
                # Alternate perspective (white's turn = positive, black's = negative)
                if i % 2 == 0:  # White's turn
                    values.append(game_result)
                else:  # Black's turn
                    values.append(-game_result)

            # Store data
            all_positions.extend(positions)
            all_policies.extend(policies)
            all_values.extend(values)

            if verbose and (game_idx + 1) % max(1, num_games // 10) == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_idx + 1) / elapsed
                eta = (num_games - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0
                result_str = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2"}[game_result]
                print(f"  Game {game_idx + 1}/{num_games} ({result_str}): "
                      f"{game_data['move_count']} moves | "
                      f"{games_per_sec:.2f} games/sec | ETA: {eta:.1f}s")

        # Convert to numpy arrays
        positions_array = np.array(all_positions, dtype=np.float32)
        policies_array = np.array(all_policies, dtype=np.float32)
        values_array = np.array(all_values, dtype=np.float32)

        if verbose:
            total_time = time.time() - start_time
            print(f"\nGenerated {num_games} games in {total_time:.1f}s")
            print(f"  Total positions: {len(all_positions):,}")
            print(f"  Avg moves per game: {len(all_positions) / num_games:.1f}")
            print(f"  White wins: {sum(1 for v in all_values[::2] if v > 0)}")
            print(f"  Black wins: {sum(1 for v in all_values[::2] if v < 0)}")
            print(f"  Draws: {sum(1 for v in all_values[::2] if v == 0)}")

        data = {
            'positions': positions_array,
            'policies': policies_array,
            'values': values_array,
            'num_games': num_games,
            'simulations_per_move': self.num_simulations
        }

        # Save if output path provided
        if output_path:
            self._save_data(data, output_path)
            if verbose:
                print(f"  Saved to: {output_path}")

        return data

    def _save_data(self, data: Dict, output_path: str):
        """Save self-play data to .npz file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            positions=data['positions'],
            policies=data['policies'],
            values=data['values'],
            num_games=data['num_games'],
            simulations_per_move=data['simulations_per_move']
        )


def test_selfplay():
    """Test self-play generation with current model."""
    print("=" * 70)
    print("Testing Self-Play Generation")
    print("=" * 70)

    # Load model
    from src.utils.improved_model import ImprovedChessModel
    from src.utils.move_mapper import MoveMapper
    import torch

    model_path = 'chess_model_best.pth'
    checkpoint = torch.load(model_path, map_location='cpu')

    model = ImprovedChessModel(hidden_size=checkpoint.get('hidden_size', 512))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    move_mapper = MoveMapper()

    # Generate a few games
    generator = SelfPlayDataGenerator(model, move_mapper, num_simulations=50)

    data = generator.generate_games(
        num_games=3,
        output_path='data/selfplay_test.npz',
        verbose=True
    )

    print("\n" + "=" * 70)
    print("Test complete! Self-play working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    test_selfplay()
