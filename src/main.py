from .utils import chess_manager, GameContext
from chess import Move
import torch
import os

# ===== SEARCH MODE CONFIGURATION =====
# Choose search mode: "mcts", "minimax", or "hybrid"
SEARCH_MODE = "hybrid"  # Options: "mcts", "minimax", "hybrid"
MINIMAX_DEPTH = 3       # Depth for minimax search (3-ply = looks 3 moves ahead)
# =====================================

# Write code here that runs once
# Load the trained neural network model and search engines
_model = None
_move_mapper = None
_mcts = None
_minimax = None
_hybrid = None

def _load_model():
    """Load the trained model and initialize search engines."""
    global _model, _move_mapper, _mcts, _minimax, _hybrid

    if _model is not None:
        return  # Already loaded

    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chess_model_best.pth')

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Falling back to random moves. Please train a model first.")
        return

    try:
        from .utils.model import ChessModel
        from .utils.improved_model import ImprovedChessModel
        from .utils.move_mapper import MoveMapper
        from .utils.mcts import MCTS
        from .utils.minimax import MinimaxSearch, HybridSearch

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Determine model type and create appropriate model
        model_type = checkpoint.get('model_type', 'ChessModel')
        hidden_size = checkpoint.get('hidden_size', 128)

        if model_type == 'ImprovedChessModel':
            print(f"Loading ImprovedChessModel (CNN-based) with hidden_size={hidden_size}")
            _model = ImprovedChessModel(hidden_size=hidden_size)
        else:
            print(f"Loading ChessModel (MLP-based)")
            num_layers = checkpoint.get('num_hidden_layers', 2)
            _model = ChessModel(hidden_size=hidden_size, num_hidden_layers=num_layers)

        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()

        # Get move mapper
        _move_mapper = checkpoint.get('move_mapper')
        if _move_mapper is None:
            _move_mapper = MoveMapper()

        # Initialize search engines based on mode
        print(f"✓ Loaded model from {model_path}")
        print(f"  Model parameters: {sum(p.numel() for p in _model.parameters()):,}")
        print(f"  Search mode: {SEARCH_MODE.upper()}")

        if SEARCH_MODE in ["mcts", "hybrid"]:
            _mcts = MCTS(_model, _move_mapper, num_simulations=200, exploration_constant=2.0)
            print(f"  MCTS initialized (200 base simulations)")

        if SEARCH_MODE in ["minimax", "hybrid"]:
            _minimax = MinimaxSearch(_model, _move_mapper, depth=MINIMAX_DEPTH)
            print(f"  Minimax initialized ({MINIMAX_DEPTH}-ply search)")

        if SEARCH_MODE == "hybrid":
            _hybrid = HybridSearch(_model, _move_mapper,
                                  mcts_simulations=200,
                                  minimax_depth=MINIMAX_DEPTH)
            print(f"  Hybrid search initialized (MCTS + Minimax verification)")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to random moves.")


# Load model on import
_load_model()


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print(f"Thinking with {SEARCH_MODE.upper()} + Neural Network...")

    # Check if game is over first
    if ctx.board.is_game_over():
        if ctx.board.is_checkmate():
            # Game is over - opponent was checkmated (we won!)
            print("Game over: Checkmate! Bot won the game.")
        elif ctx.board.is_stalemate():
            print("Game over: Stalemate.")
        else:
            print("Game over: Draw or other termination.")
        ctx.logProbabilities({})
        raise ValueError("Game is already over - no moves available")

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Use selected search engine based on mode
    if SEARCH_MODE == "mcts" and _mcts is not None:
        # Improved time management
        time_left_ms = ctx.timeLeft
        move_number = len(list(ctx.board.move_stack)) + 1

        # Opening: fewer simulations (faster)
        if move_number <= 10:
            if time_left_ms > 60000:
                _mcts.num_simulations = 300
            elif time_left_ms > 30000:
                _mcts.num_simulations = 200
            else:
                _mcts.num_simulations = 100
        # Middle game: more simulations
        elif move_number <= 30:
            if time_left_ms > 60000:
                _mcts.num_simulations = 500  # Increased!
            elif time_left_ms > 30000:
                _mcts.num_simulations = 350
            elif time_left_ms > 10000:
                _mcts.num_simulations = 250
            else:
                _mcts.num_simulations = 150
        # Endgame: even more simulations (critical positions)
        else:
            if time_left_ms > 60000:
                _mcts.num_simulations = 600  # Even more for endgame!
            elif time_left_ms > 30000:
                _mcts.num_simulations = 400
            elif time_left_ms > 10000:
                _mcts.num_simulations = 300
            else:
                _mcts.num_simulations = 200

        # Run MCTS search
        best_move, move_probs = _mcts.search(ctx.board)

        # Log probabilities
        ctx.logProbabilities(move_probs)

        return best_move

    elif SEARCH_MODE == "minimax" and _minimax is not None:
        # Pure minimax search
        # Adjust depth based on time and position complexity
        time_left_ms = ctx.timeLeft
        move_number = len(list(ctx.board.move_stack)) + 1

        # Determine search depth based on time
        if time_left_ms > 60000:
            depth = min(MINIMAX_DEPTH + 1, 5)  # Extra depth if we have time
        elif time_left_ms > 30000:
            depth = MINIMAX_DEPTH
        elif time_left_ms > 10000:
            depth = max(MINIMAX_DEPTH - 1, 2)
        else:
            depth = 2  # Minimum depth

        # Run minimax search
        best_move, move_probs = _minimax.search(ctx.board, depth=depth)

        # Log probabilities
        ctx.logProbabilities(move_probs)

        print(f"  Minimax searched {_minimax.nodes_searched} nodes, pruned {_minimax.pruned_branches} branches")

        return best_move

    elif SEARCH_MODE == "hybrid" and _hybrid is not None:
        # Hybrid MCTS + Minimax search
        time_left_ms = ctx.timeLeft
        move_number = len(list(ctx.board.move_stack)) + 1

        # Adjust MCTS simulations (same as pure MCTS)
        if move_number <= 10:
            if time_left_ms > 60000:
                _hybrid.num_simulations = 300
            elif time_left_ms > 30000:
                _hybrid.num_simulations = 200
            else:
                _hybrid.num_simulations = 100
        elif move_number <= 30:
            if time_left_ms > 60000:
                _hybrid.num_simulations = 500
            elif time_left_ms > 30000:
                _hybrid.num_simulations = 350
            elif time_left_ms > 10000:
                _hybrid.num_simulations = 250
            else:
                _hybrid.num_simulations = 150
        else:
            if time_left_ms > 60000:
                _hybrid.num_simulations = 600
            elif time_left_ms > 30000:
                _hybrid.num_simulations = 400
            elif time_left_ms > 10000:
                _hybrid.num_simulations = 300
            else:
                _hybrid.num_simulations = 200

        # Run hybrid search
        best_move, move_probs = _hybrid.search(ctx.board)

        # Log probabilities
        ctx.logProbabilities(move_probs)

        return best_move

    else:
        # Fallback to random if model not loaded
        import random
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choice(legal_moves)


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    # MCTS doesn't need reset as it creates a new tree each search
    pass
