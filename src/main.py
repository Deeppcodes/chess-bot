from .utils import chess_manager, GameContext
from chess import Move
import torch
import os

# Write code here that runs once
# Load the trained neural network model and MCTS
_model = None
_move_mapper = None
_mcts = None

def _load_model():
    """Load the trained model and initialize MCTS."""
    global _model, _move_mapper, _mcts
    
    if _model is not None:
        return  # Already loaded
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'chess_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Falling back to random moves. Please train a model first.")
        return
    
    try:
        from .utils.model import ChessModel
        from .utils.improved_model import ImprovedChessModel
        from .utils.move_mapper import MoveMapper
        from .utils.mcts import MCTS
        
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
        
        # Create MCTS
        # Adjust number of simulations based on available time
        # Default to 50 simulations, but can be adjusted
        _mcts = MCTS(_model, _move_mapper, num_simulations=50, exploration_constant=1.5)
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Model parameters: {sum(p.numel() for p in _model.parameters()):,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to random moves.")


# Load model on import
_load_model()


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Thinking with MCTS + Neural Network...")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Use MCTS + Neural Network if available, otherwise fall back to random
    if _mcts is not None:
        # Adjust simulations based on available time
        # More time = more simulations = better moves
        time_left_ms = ctx.timeLeft
        if time_left_ms > 30000:  # More than 30 seconds
            _mcts.num_simulations = 100
        elif time_left_ms > 10000:  # More than 10 seconds
            _mcts.num_simulations = 50
        else:  # Less time
            _mcts.num_simulations = 25
        
        # Run MCTS search
        best_move, move_probs = _mcts.search(ctx.board)
        
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
