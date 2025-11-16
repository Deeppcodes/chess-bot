"""
Minimax search with alpha-beta pruning and neural network evaluation.
Provides deeper tactical search compared to MCTS.
"""

import math
from typing import List, Dict, Tuple, Optional
from chess import Board, Move
import torch


class MinimaxSearch:
    """
    Minimax search with alpha-beta pruning.
    Uses neural network for position evaluation.
    """

    def __init__(self, model, move_mapper, depth: int = 3):
        """
        Initialize Minimax search.

        Args:
            model: Trained neural network model
            move_mapper: MoveMapper instance
            depth: Search depth (default: 3 = 3-ply)
        """
        self.model = model
        self.move_mapper = move_mapper
        self.depth = depth
        self.model.eval()

        # Statistics
        self.nodes_searched = 0
        self.pruned_branches = 0

        # Piece values for move ordering
        self.piece_values = {
            1: 1,   # Pawn
            2: 3,   # Knight
            3: 3,   # Bishop
            4: 5,   # Rook
            5: 9,   # Queen
            6: 0    # King
        }

    def search(self, board: Board, depth: Optional[int] = None) -> Tuple[Move, Dict[Move, float]]:
        """
        Perform minimax search from given position.

        Args:
            board: Current board position
            depth: Search depth (uses default if None)

        Returns:
            Best move and move evaluations
        """
        if depth is None:
            depth = self.depth

        # Reset statistics
        self.nodes_searched = 0
        self.pruned_branches = 0

        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            return None, {}

        # Order moves for better pruning (captures first, then checks)
        ordered_moves = self._order_moves(board, legal_moves)

        best_move = None
        best_score = float('-inf')
        move_scores = {}

        alpha = float('-inf')
        beta = float('inf')

        # Search each move
        for move in ordered_moves:
            board.push(move)

            # Recursively search
            score = -self._minimax(board, depth - 1, -beta, -alpha, False)

            board.pop()

            move_scores[move] = score

            if score > best_score:
                best_score = score
                best_move = move

            # Alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                self.pruned_branches += 1
                break

        # Convert scores to probabilities for logging
        move_probs = self._scores_to_probabilities(move_scores, legal_moves)

        return best_move, move_probs

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float,
                 is_maximizing: bool) -> float:
        """
        Minimax search with alpha-beta pruning.

        Args:
            board: Current position
            depth: Remaining depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: True if maximizing player

        Returns:
            Evaluation score
        """
        self.nodes_searched += 1

        # Terminal conditions
        if board.is_checkmate():
            # Checkmate is very bad (or very good if we're checkmating)
            return -10000 if is_maximizing else 10000

        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0  # Draw

        # Base case: evaluate position with neural network
        if depth == 0:
            return self._evaluate_position(board)

        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            return 0  # Draw (stalemate)

        # Order moves for better pruning
        ordered_moves = self._order_moves(board, legal_moves)

        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    self.pruned_branches += 1
                    break  # Beta cutoff

            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    self.pruned_branches += 1
                    break  # Alpha cutoff

            return min_eval

    def _evaluate_position(self, board: Board) -> float:
        """
        Evaluate position using neural network.

        Args:
            board: Board position to evaluate

        Returns:
            Evaluation score from current player's perspective
        """
        from .board_encoder import board_to_tensor_torch

        board_tensor = board_to_tensor_torch(board)

        with torch.no_grad():
            _, value = self.model(board_tensor)

        # Get value estimate
        value_estimate = value.item()

        # Model outputs from white's perspective
        # Adjust based on whose turn it is
        if not board.turn:  # Black's turn
            value_estimate = -value_estimate

        return value_estimate

    def _order_moves(self, board: Board, moves: List[Move]) -> List[Move]:
        """
        Order moves for better alpha-beta pruning.
        Priority: Checkmates > Captures (MVV-LVA) > Checks > Others

        Args:
            board: Current board position
            moves: List of legal moves

        Returns:
            Ordered list of moves
        """
        move_scores = []

        for move in moves:
            score = 0

            # Check for checkmate (highest priority)
            test_board = board.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                score += 100000
            elif test_board.is_check():
                score += 10000  # Checks are high priority

            # Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    captured_value = self.piece_values.get(captured_piece.piece_type, 0)
                    attacking_piece = board.piece_at(move.from_square)
                    attacker_value = self.piece_values.get(attacking_piece.piece_type, 0) if attacking_piece else 0

                    # MVV-LVA: Prefer capturing valuable pieces with cheap pieces
                    score += captured_value * 100 - attacker_value

                    # Extra bonus if square is not defended
                    if not test_board.is_attacked_by(not board.turn, move.to_square):
                        score += 5000  # Free capture!

            # Promotion
            if move.promotion:
                score += 8000  # Promotions are valuable

            move_scores.append((move, score))

        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in move_scores]

    def _scores_to_probabilities(self, move_scores: Dict[Move, float],
                                 legal_moves: List[Move]) -> Dict[Move, float]:
        """
        Convert move scores to probabilities for logging.

        Args:
            move_scores: Dictionary mapping moves to scores
            legal_moves: All legal moves

        Returns:
            Dictionary mapping moves to probabilities
        """
        if not move_scores:
            return {move: 1.0 / len(legal_moves) for move in legal_moves}

        # Apply softmax to scores
        # First, shift scores to avoid overflow (subtract max)
        max_score = max(move_scores.values())
        exp_scores = {}

        for move, score in move_scores.items():
            # Use temperature to control sharpness
            temperature = 0.5  # Lower = sharper distribution
            exp_scores[move] = math.exp((score - max_score) / temperature)

        total = sum(exp_scores.values())

        if total == 0:
            return {move: 1.0 / len(legal_moves) for move in legal_moves}

        # Normalize
        move_probs = {}
        for move in legal_moves:
            if move in exp_scores:
                move_probs[move] = exp_scores[move] / total
            else:
                move_probs[move] = 0.0

        return move_probs


class HybridSearch:
    """
    Hybrid search combining MCTS and Minimax.
    Uses minimax for tactical verification of MCTS moves.
    """

    def __init__(self, model, move_mapper, mcts_simulations: int = 200,
                 minimax_depth: int = 3):
        """
        Initialize hybrid search.

        Args:
            model: Trained neural network model
            move_mapper: MoveMapper instance
            mcts_simulations: Number of MCTS simulations
            minimax_depth: Minimax search depth
        """
        from .mcts import MCTS

        self.mcts = MCTS(model, move_mapper, num_simulations=mcts_simulations)
        self.minimax = MinimaxSearch(model, move_mapper, depth=minimax_depth)

        # Allow dynamic adjustment
        self.num_simulations = mcts_simulations

    def search(self, board: Board) -> Tuple[Move, Dict[Move, float]]:
        """
        Perform hybrid search.
        Uses MCTS for broad search, minimax for tactical verification.

        Args:
            board: Current board position

        Returns:
            Best move and move probabilities
        """
        # Update MCTS simulation count
        self.mcts.num_simulations = self.num_simulations

        # Get top candidates from MCTS
        mcts_move, mcts_probs = self.mcts.search(board)

        # Get top 3 moves from MCTS
        sorted_moves = sorted(mcts_probs.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [move for move, _ in sorted_moves[:3]]

        # Verify with minimax (shallow search for speed)
        best_move = None
        best_score = float('-inf')

        for move in top_candidates:
            board.push(move)
            # Quick minimax verification (depth-1 for speed)
            score = -self.minimax._minimax(board, self.minimax.depth - 1,
                                          float('-inf'), float('inf'), False)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        # Use MCTS probabilities for logging
        return best_move if best_move else mcts_move, mcts_probs
