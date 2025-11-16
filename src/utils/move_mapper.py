"""
Move mapping utilities to convert between chess moves and model output indices.
Uses unified move encoding with 4096 output dimensions.
"""

from chess import Move, Square
from typing import Dict, List, Optional
import numpy as np
from .move_encoder import MoveEncoder


class MoveMapper:
    """
    Maps chess moves to model output indices.
    Uses unified encoding: from_square * 64 + to_square (4096 dimensions).
    This is a wrapper around MoveEncoder for backward compatibility.
    """

    def __init__(self):
        """Initialize with MoveEncoder."""
        self.encoder = MoveEncoder()
        self.max_indices = self.encoder.num_outputs  # 4096

    def get_move_index(self, move: Move) -> int:
        """
        Get the index for a move.

        Args:
            move: Chess move (can be Move object or UCI string)

        Returns:
            Index in range [0, 4095]
        """
        return self.encoder.encode_move(move)

    def create_policy_target(self, legal_moves: List[Move], target_move: Move) -> np.ndarray:
        """
        Create a policy target vector (one-hot for the target move).

        Args:
            legal_moves: List of legal moves (unused, kept for compatibility)
            target_move: The move that was actually played

        Returns:
            Array of shape (4096,) with 1.0 at target move index, 0.0 elsewhere
        """
        return self.encoder.create_policy_target(target_move, as_index=False)

    def get_move_probabilities(self, policy_logits: np.ndarray, legal_moves: List[Move]) -> Dict[Move, float]:
        """
        Convert model policy output to move probabilities for legal moves only.

        Args:
            policy_logits: Raw logits from model, shape (4096,)
            legal_moves: List of legal moves in current position

        Returns:
            Dictionary mapping moves to probabilities
        """
        return self.encoder.get_move_probabilities(policy_logits, legal_moves)

