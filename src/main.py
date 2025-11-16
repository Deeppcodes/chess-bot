"""
Lightweight ChessHacks deployment version.
Uses minimax with static evaluation instead of neural network.
This version works without PyTorch!
"""

from .utils import chess_manager, GameContext
from chess import Move
import random

# Simple piece values for evaluation
PIECE_VALUES = {
    1: 100,   # Pawn
    2: 320,   # Knight
    3: 330,   # Bishop
    4: 500,   # Rook
    5: 900,   # Queen
    6: 20000  # King
}


def evaluate_board(board):
    """
    Simple material-based evaluation.
    Positive = good for white, negative = good for black.
    """
    if board.is_checkmate():
        return -20000 if board.turn else 20000

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    # Material count
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES.get(piece.piece_type, 0)
            score += value if piece.color else -value

    # Mobility bonus (number of legal moves)
    mobility = len(list(board.legal_moves))
    score += mobility * 10 if board.turn else -mobility * 10

    return score


def minimax(board, depth, alpha, beta, maximizing):
    """
    Minimax with alpha-beta pruning.
    Fast enough to run on ChessHacks without GPU.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)

    # Move ordering: captures first
    legal_moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main bot logic - uses lightweight minimax search.
    No neural network needed!
    """
    print("Thinking with minimax (lightweight deployment)...")

    if ctx.board.is_game_over():
        ctx.logProbabilities({})
        raise ValueError("Game is already over")

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Determine search depth based on time
    time_left_ms = ctx.timeLeft
    if time_left_ms > 60000:
        depth = 4
    elif time_left_ms > 30000:
        depth = 3
    elif time_left_ms > 10000:
        depth = 2
    else:
        depth = 1

    # Find best move with minimax
    best_move = None
    best_score = float('-inf') if ctx.board.turn else float('inf')

    # Order moves: checks and captures first
    def move_priority(move):
        is_capture = ctx.board.is_capture(move)
        test_board = ctx.board.copy()
        test_board.push(move)
        is_check = test_board.is_check()
        return (is_capture, is_check)

    ordered_moves = sorted(legal_moves, key=move_priority, reverse=True)

    move_scores = {}

    for move in ordered_moves:
        board_copy = ctx.board.copy()
        board_copy.push(move)

        score = minimax(board_copy, depth - 1,
                       float('-inf'), float('inf'),
                       not ctx.board.turn)

        move_scores[move] = score

        if ctx.board.turn:  # White
            if score > best_score:
                best_score = score
                best_move = move
        else:  # Black
            if score < best_score:
                best_score = score
                best_move = move

    # Convert scores to probabilities for logging
    if move_scores:
        # Normalize scores to probabilities
        min_score = min(move_scores.values())
        adjusted_scores = {m: s - min_score + 1 for m, s in move_scores.items()}
        total = sum(adjusted_scores.values())
        move_probs = {m: s / total for m, s in adjusted_scores.items()}
    else:
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}

    ctx.logProbabilities(move_probs)

    return best_move if best_move else legal_moves[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Reset function (no state to clear)."""
    pass
