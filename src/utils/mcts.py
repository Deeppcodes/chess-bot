"""
Monte Carlo Tree Search (MCTS) implementation with neural network guidance.
Uses neural network for policy (move probabilities) and value (position evaluation).
"""

import math
import numpy as np
from typing import List, Optional, Dict, Tuple
from chess import Board, Move
import torch


class MCTSNode:
    """Node in the MCTS search tree."""
    
    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None):
        """
        Initialize a MCTS node.
        
        Args:
            board: Current board position
            parent: Parent node (None for root)
            move: Move that led to this position
        """
        self.board = board
        self.parent = parent
        self.move = move
        
        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.value_estimate = 0.0  # From neural network
        
        # Children
        self.children: Dict[Move, 'MCTSNode'] = {}
        self.legal_moves = list(board.generate_legal_moves())
        
        # Policy prior from neural network (probability for each legal move)
        self.policy_prior: Dict[Move, float] = {}
        
        # Whether this node has been expanded
        self.expanded = False
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been explored."""
        return len(self.children) == len(self.legal_moves) and self.expanded
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.board.is_game_over()
    
    def get_value(self) -> float:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return self.value_estimate
        return self.value_sum / self.visit_count
    
    def ucb_score(self, exploration_constant: float = 1.5) -> float:
        """
        Calculate UCB1 score for selection.
        Only valid for non-root nodes.
        """
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.get_value()
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def select_child(self, exploration_constant: float = 1.5) -> 'MCTSNode':
        """
        Select child using PUCT (Policy + UCT) algorithm.
        Improved version with better numerical stability.
        """
        best_score = float('-inf')
        best_child_move = None
        parent_visits = self.visit_count + 1  # +1 for numerical stability

        for move in self.legal_moves:
            prior = self.policy_prior.get(move, 1.0 / len(self.legal_moves))

            if move not in self.children:
                # Unexplored move - high priority for exploration
                # Use prior * sqrt(parent_visits) to encourage exploration
                score = prior * math.sqrt(parent_visits)
            else:
                child = self.children[move]
                q_value = child.get_value()

                # Improved PUCT formula: Q + c_puct * P * sqrt(sum(N)) / (1 + n)
                # This is more standard and numerically stable
                puct_term = (exploration_constant * prior *
                            math.sqrt(parent_visits) / (1 + child.visit_count))
                score = q_value + puct_term

            if score > best_score:
                best_score = score
                best_child_move = move

        if best_child_move is None:
            best_child_move = self.legal_moves[0] if self.legal_moves else None
            if best_child_move is None:
                return None

        # Create child if doesn't exist
        if best_child_move not in self.children:
            child_board = self.board.copy()
            child_board.push(best_child_move)
            child_node = MCTSNode(child_board, parent=self, move=best_child_move)
            self.children[best_child_move] = child_node

        return self.children[best_child_move]
    
    def expand(self, policy_prior: Dict[Move, float], value_estimate: float):
        """
        Expand this node by adding policy prior and value estimate.
        
        Args:
            policy_prior: Dictionary mapping moves to probabilities
            value_estimate: Value estimate from neural network
        """
        self.policy_prior = policy_prior
        self.value_estimate = value_estimate
        self.expanded = True
    
    def backpropagate(self, value: float):
        """
        Backpropagate value up the tree.
        
        Args:
            value: Value to propagate (from perspective of node's player)
        """
        self.visit_count += 1
        self.value_sum += value
        
        # Propagate to parent (negate value since it's opponent's perspective)
        if self.parent is not None:
            self.parent.backpropagate(-value)
    
    def get_best_move(self, use_value_weight: bool = True) -> Move:
        """
        Get the best move using visit count, optionally weighted by value.

        Args:
            use_value_weight: If True, weight visits by average value
        """
        if not self.children:
            return self.legal_moves[0] if self.legal_moves else None

        if use_value_weight:
            # Combine visit count with average value
            # Moves with high visits AND good outcomes are best
            best_move = max(self.children.items(),
                           key=lambda x: x[1].visit_count * (1 + x[1].get_value()))[0]
        else:
            # Original: just visit count
            best_move = max(self.children.items(),
                           key=lambda x: x[1].visit_count)[0]

        return best_move


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """
    
    def __init__(self, model, move_mapper, num_simulations: int = 100, exploration_constant: float = 1.5):
        """
        Initialize MCTS.
        
        Args:
            model: Trained neural network model
            move_mapper: MoveMapper instance
            num_simulations: Number of MCTS simulations per move
            exploration_constant: Exploration constant for UCB
        """
        self.model = model
        self.move_mapper = move_mapper
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.model.eval()
    
    def search(self, board: Board) -> Tuple[Move, Dict[Move, float]]:
        """
        Perform MCTS search from given position.
        
        Args:
            board: Current board position
            
        Returns:
            Best move and move probabilities (from visit counts)
        """
        root = MCTSNode(board)
        
        # Get initial policy and value from neural network
        policy_prior, value_estimate = self._evaluate_position(root.board)
        root.expand(policy_prior, value_estimate)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Selection: traverse from root to leaf
            node = self._select(root)
            
            # Evaluation: get policy and value from neural network
            if not node.is_terminal():
                policy_prior, value_estimate = self._evaluate_position(node.board)
                node.expand(policy_prior, value_estimate)
                
                # Expand: add children for all legal moves
                self._expand(node)
            
            # Backpropagation: update statistics
            if node.is_terminal():
                # Terminal node: get actual game result
                value = self._get_terminal_value(node.board)
            else:
                # Use neural network value estimate
                value = value_estimate
            
            node.backpropagate(value)
        
        # Get best move and probabilities
        best_move = root.get_best_move(use_value_weight=True)  # Better selection!

        # Add temperature scaling for move probabilities
        # Lower temperature = more deterministic (use best move)
        # Higher temperature = more exploratory
        temperature = 1.0  # Can tune this (0.5 = more deterministic, 1.5 = more exploratory)
        move_probs = self._get_move_probabilities(root, temperature=temperature)

        return best_move, move_probs
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Select a leaf node by traversing the tree.
        
        Args:
            root: Root node
            
        Returns:
            Leaf node to evaluate
        """
        node = root
        
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child(self.exploration_constant)
            if node is None:
                break
        
        return node
    
    def _expand(self, node: MCTSNode):
        """
        Expand a node by creating children for all legal moves.
        
        Args:
            node: Node to expand
        """
        for move in node.legal_moves:
            if move not in node.children:
                # Create child node
                child_board = node.board.copy()
                child_board.push(move)
                child_node = MCTSNode(child_board, parent=node, move=move)
                node.children[move] = child_node
    
    def _evaluate_position(self, board: Board) -> Tuple[Dict[Move, float], float]:
        """
        Evaluate position using neural network.

        Args:
            board: Board position to evaluate

        Returns:
            Tuple of (policy_prior, value_estimate)
        """
        from .board_encoder import board_to_tensor_torch

        board_tensor = board_to_tensor_torch(board)
        legal_moves = list(board.generate_legal_moves())

        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)

        # Convert to move probabilities
        policy_prior = self.move_mapper.get_move_probabilities(
            policy_logits[0].numpy(), legal_moves
        )

        # TACTICAL BOOST: Increase probability for capturing hanging pieces
        policy_prior = self._apply_tactical_boost(board, policy_prior, legal_moves)

        # Get value estimate
        value_estimate = value.item()

        # Adjust value based on whose turn it is
        # Model outputs from white's perspective
        if not board.turn:  # Black's turn
            value_estimate = -value_estimate

        return policy_prior, value_estimate

    def _apply_tactical_boost(self, board: Board, policy_prior: Dict[Move, float],
                               legal_moves: List[Move], boost_factor: float = 3.0) -> Dict[Move, float]:
        """
        Enhanced tactical boost with checkmate detection and better evaluation.

        Args:
            board: Current board position
            policy_prior: Original policy from neural network
            legal_moves: List of legal moves
            boost_factor: Multiplier for tactical moves (default: 3.0)

        Returns:
            Adjusted policy with tactical boosts
        """
        piece_values = {
            1: 1,   # Pawn
            2: 3,   # Knight
            3: 3,   # Bishop
            4: 5,   # Rook
            5: 9,   # Queen
            6: 0    # King
        }

        tactical_scores = {}

        # FIRST: Check for immediate checkmates (highest priority!)
        for move in legal_moves:
            test_board = board.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                tactical_scores[move] = 100.0  # MASSIVE boost for checkmate!
                continue  # Skip other checks, checkmate is best

        # SECOND: Check for checks (high priority)
        for move in legal_moves:
            if move in tactical_scores:  # Already found checkmate
                continue

            test_board = board.copy()
            test_board.push(move)
            if test_board.is_check():
                # Check is valuable, but less than checkmate
                tactical_scores[move] = 2.0

        # THIRD: Evaluate captures and piece safety
        for move in legal_moves:
            if move in tactical_scores:  # Already scored
                continue

            score = 0.0

            # Captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    captured_value = piece_values.get(captured_piece.piece_type, 0)
                    attacking_piece = board.piece_at(move.from_square)
                    attacker_value = piece_values.get(attacking_piece.piece_type, 0) if attacking_piece else 0

                    test_board = board.copy()
                    test_board.push(move)

                    # Check if square is defended after capture
                    if test_board.is_attacked_by(not board.turn, move.to_square):
                        # Defended - only good if trading up or equal
                        if captured_value >= attacker_value:
                            score += captured_value * 0.5
                        elif captured_value < attacker_value:
                            score -= (attacker_value - captured_value) * 0.3  # Penalty for bad trade
                    else:
                        # Hanging piece! Very valuable
                        score += captured_value * 2.5  # Increased from 2.0

                        # Extra boost for winning trades
                        if captured_value > attacker_value:
                            score += (captured_value - attacker_value) * 2.0  # Increased from 1.5

            # Check if move attacks opponent's pieces
            test_board = board.copy()
            test_board.push(move)
            from_square = move.to_square  # Square we moved to

            # Check what pieces this square attacks
            if test_board.is_attacked_by(board.turn, from_square):
                attacked_squares = [sq for sq in range(64)
                                  if test_board.is_attacked_by(board.turn, sq)
                                  and test_board.piece_at(sq)
                                  and test_board.piece_at(sq).color != board.turn]
                if attacked_squares:
                    # We're attacking something - small bonus
                    score += 0.3

            tactical_scores[move] = score

        # Apply boosts
        boosted_policy = {}
        for move in legal_moves:
            original_prob = policy_prior.get(move, 0.0)
            tactical_score = tactical_scores.get(move, 0.0)

            if tactical_score > 0:
                # Exponential boost for very high scores (like checkmate)
                if tactical_score >= 50:
                    boosted_policy[move] = original_prob * (1.0 + tactical_score)
                else:
                    boosted_policy[move] = original_prob * (1.0 + boost_factor * tactical_score)
            else:
                boosted_policy[move] = max(0.0, original_prob * (1.0 + tactical_score))  # Allow penalties

        # Renormalize
        total_prob = sum(boosted_policy.values())
        if total_prob > 0:
            boosted_policy = {move: prob / total_prob for move, prob in boosted_policy.items()}
        else:
            # Fallback to uniform if all probabilities went to zero
            boosted_policy = {move: 1.0 / len(legal_moves) for move in legal_moves}

        return boosted_policy

    def _quiescence_search(self, board: Board, depth: int = 3, alpha: float = -float('inf'),
                           beta: float = float('inf')) -> float:
        """
        Quiescence search to evaluate tactical lines more deeply.
        Only searches captures and checks after the main search.

        Args:
            board: Current position
            depth: Maximum depth for quiescence
            alpha: Alpha for alpha-beta pruning
            beta: Beta for alpha-beta pruning

        Returns:
            Evaluation of position
        """
        if depth <= 0:
            # Base case: evaluate with neural network
            policy_prior, value_estimate = self._evaluate_position(board)
            return value_estimate if board.turn else -value_estimate

        # Get stand-pat evaluation
        policy_prior, stand_pat = self._evaluate_position(board)
        stand_pat = stand_pat if board.turn else -stand_pat

        # Alpha-beta pruning
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate tactical moves (captures and checks)
        tactical_moves = []
        for move in board.generate_legal_moves():
            test_board = board.copy()
            test_board.push(move)
            if board.is_capture(move) or test_board.is_check():
                tactical_moves.append(move)

        # Search tactical moves
        for move in tactical_moves:
            test_board = board.copy()
            test_board.push(move)

            score = -self._quiescence_search(test_board, depth - 1, -beta, -alpha)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _get_terminal_value(self, board: Board) -> float:
        """
        Get value for terminal position.
        
        Args:
            board: Terminal board position
            
        Returns:
            Value from perspective of player to move
        """
        if board.is_checkmate():
            return -1.0  # Lost (opponent checkmated us)
        else:
            return 0.0  # Draw
    
    def _get_move_probabilities(self, root: MCTSNode, temperature: float = 1.0) -> Dict[Move, float]:
        """
        Get move probabilities from visit counts with temperature scaling.

        Args:
            root: Root node
            temperature: Temperature for scaling (1.0 = normal, <1.0 = sharper, >1.0 = softer)

        Returns:
            Dictionary mapping moves to probabilities
        """
        if not root.children:
            return {move: 1.0 / len(root.legal_moves) for move in root.legal_moves}

        # Get visit counts
        visit_counts = {}
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count ** (1.0 / temperature)

        total_visits = sum(visit_counts.values())

        if total_visits == 0:
            return {move: 1.0 / len(root.legal_moves) for move in root.legal_moves}

        move_probs = {}
        for move in root.legal_moves:
            if move in visit_counts:
                move_probs[move] = visit_counts[move] / total_visits
            else:
                move_probs[move] = 0.0

        # Renormalize
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}

        return move_probs

