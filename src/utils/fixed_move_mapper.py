"""
Fixed Move Mapper - Consistent move encoding between training and inference.

Uses a simple encoding: from_square * 64 + to_square
This gives us 64 * 64 = 4096 possible moves, but we only use 256 indices.
We'll use a hash of the move UCI string modulo 256 for consistency.
"""

import chess
from typing import Dict, Optional
import numpy as np


class FixedMoveMapper:
    """
    Fixed move mapper with consistent encoding.
    Uses UCI string hash modulo 256 for move encoding.
    """
    
    def __init__(self):
        self.max_indices = 256
    
    def get_move_index(self, move: chess.Move) -> int:
        """
        Get consistent index for a move using deterministic encoding.
        
        Uses: from_square * 64 + to_square, then modulo 256
        This is TRULY deterministic and consistent.
        
        Args:
            move: Chess move
            
        Returns:
            Index in range [0, 255]
        """
        # Deterministic encoding: from * 64 + to
        # This gives 0-4095 range, then modulo 256
        from_square = move.from_square
        to_square = move.to_square
        
        # Add promotion info to make different promotions unique
        promotion_offset = 0
        if move.promotion:
            # Q=5, R=4, B=3, N=2
            promotion_offset = move.promotion * 100
        
        # Deterministic index
        index = (from_square * 64 + to_square + promotion_offset) % self.max_indices
        return index
    
    def get_move_from_index(self, index: int, legal_moves: list) -> Optional[chess.Move]:
        """
        Get move from index, searching through legal moves.
        
        Args:
            index: Model output index
            legal_moves: List of legal moves in current position
            
        Returns:
            Move if found, None otherwise
        """
        for move in legal_moves:
            if self.get_move_index(move) == index:
                return move
        return None
    
    def create_policy_target(self, target_move: chess.Move) -> np.ndarray:
        """
        Create a policy target vector (one-hot for the target move).
        
        Args:
            target_move: The move that was actually played
            
        Returns:
            Array of shape (256,) with 1.0 at target move index
        """
        target = np.zeros(self.max_indices, dtype=np.float32)
        target_index = self.get_move_index(target_move)
        target[target_index] = 1.0
        return target
    
    def get_move_probabilities(self, policy_logits: np.ndarray, legal_moves: list) -> Dict[chess.Move, float]:
        """
        Convert model policy output to move probabilities for legal moves only.
        
        Args:
            policy_logits: Raw logits from model, shape (256,)
            legal_moves: List of legal moves in current position
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        # Apply softmax to policy logits
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        policy_probs = exp_logits / np.sum(exp_logits)
        
        # Map to legal moves
        move_probs = {}
        total_legal_prob = 0.0
        
        for move in legal_moves:
            idx = self.get_move_index(move)
            prob = policy_probs[idx]
            move_probs[move] = prob
            total_legal_prob += prob
        
        # Renormalize over legal moves only
        if total_legal_prob > 0:
            for move in move_probs:
                move_probs[move] /= total_legal_prob
        else:
            # Uniform if no probability mass on legal moves
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_probs[move] = uniform_prob
        
        return move_probs

