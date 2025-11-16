"""
Unified move encoding for chess moves.
Handles all legal chess moves including promotions consistently across training and inference.

Encoding scheme:
- Regular moves: from_square * 64 + to_square (0-4095)
- Total output dimension: 4096

This is a simplified encoding that works for most moves. For underpromotions
(promote to rook/bishop/knight), we use the same index as queen promotion.
This is acceptable because underpromotions are rare in practice.
"""

from chess import Move
import numpy as np
from typing import List, Dict


class MoveEncoder:
    """
    Encodes chess moves to indices for neural network training and inference.
    Uses consistent encoding: from_square * 64 + to_square
    """

    def __init__(self):
        """Initialize move encoder with 4096 output dimensions."""
        self.num_outputs = 4096

    def encode_move(self, move: Move) -> int:
        """
        Encode a chess move to an integer index.

        Args:
            move: Chess move (python-chess Move object or UCI string)

        Returns:
            Index in range [0, 4095]
        """
        if isinstance(move, str):
            move = Move.from_uci(move)

        # Simple encoding: from_square * 64 + to_square
        index = move.from_square * 64 + move.to_square

        # Ensure it's within bounds
        index = index % self.num_outputs

        return index

    def encode_move_uci(self, move_uci: str) -> int:
        """
        Encode a UCI move string to an integer index.

        Args:
            move_uci: Move in UCI format (e.g., "e2e4", "e7e8q")

        Returns:
            Index in range [0, 4095]
        """
        move = Move.from_uci(move_uci)
        return self.encode_move(move)

    def get_legal_move_indices(self, legal_moves: List[Move]) -> List[int]:
        """
        Get indices for all legal moves.

        Args:
            legal_moves: List of legal chess moves

        Returns:
            List of indices corresponding to the legal moves
        """
        return [self.encode_move(move) for move in legal_moves]

    def get_move_probabilities(self, policy_logits: np.ndarray, legal_moves: List[Move]) -> Dict[Move, float]:
        """
        Convert model policy output to move probabilities for legal moves only.

        Args:
            policy_logits: Raw logits from model, shape (4096,)
            legal_moves: List of legal moves in current position

        Returns:
            Dictionary mapping moves to probabilities
        """
        # Get indices for legal moves
        legal_indices = self.get_legal_move_indices(legal_moves)

        # Extract logits for legal moves
        legal_logits = policy_logits[legal_indices]

        # Apply softmax to get probabilities
        exp_logits = np.exp(legal_logits - np.max(legal_logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Create dictionary
        move_probs = {move: float(prob) for move, prob in zip(legal_moves, probabilities)}
        return move_probs

    def get_best_move(self, policy_logits: np.ndarray, legal_moves: List[Move]) -> Move:
        """
        Get the best legal move according to the policy.

        Args:
            policy_logits: Raw logits from model, shape (4096,)
            legal_moves: List of legal moves in current position

        Returns:
            Best legal move
        """
        move_probs = self.get_move_probabilities(policy_logits, legal_moves)
        return max(move_probs.items(), key=lambda x: x[1])[0]

    def create_policy_target(self, target_move: Move, as_index: bool = True):
        """
        Create a policy target for training.

        Args:
            target_move: The move that was actually played
            as_index: If True, return index; if False, return one-hot vector

        Returns:
            Either index (int) or one-hot array of shape (4096,)
        """
        index = self.encode_move(target_move)

        if as_index:
            return index
        else:
            # One-hot encoding
            target = np.zeros(self.num_outputs, dtype=np.float32)
            target[index] = 1.0
            return target
