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
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chess_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Falling back to random moves. Please train a model first.")
        return
    
    try:
        from .utils.model import ChessModel
        from .utils.fixed_move_mapper import FixedMoveMapper  # FIXED MAPPER!
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create new CNN/ResNet model (replaces old MLP architecture)
        # Default: 6 residual blocks, 64 channels
        # You can adjust these parameters if needed
        _model = ChessModel(num_residual_blocks=6, channels=64, dropout=0.0)
        
        # Load state dict with strict=False to handle architecture mismatches
        # This allows graceful loading of old MLP checkpoints (weights will be skipped)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect if this is an old MLP checkpoint
        is_old_model = not any('conv' in key or 'residual' in key for key in state_dict.keys())
        
        try:
            _model.load_state_dict(state_dict, strict=False)
            if is_old_model:
                print("  [WARNING] Old MLP checkpoint detected - weights not compatible, using new CNN architecture")
                print("  Please retrain the model with the new architecture for best performance")
            else:
                print("  [OK] Model weights loaded successfully")
        except Exception as load_error:
            print(f"  Warning: Could not load weights: {load_error}")
            print("  Model initialized with random weights - please retrain")
        
        _model.eval()
        
        # Use FIXED move mapper (consistent encoding between training and inference!)
        _move_mapper = FixedMoveMapper()
        
        print(f"[OK] Loaded model from {model_path}")
        print(f"  Model architecture: CNN/ResNet")
        print(f"  Move mapper: FixedMoveMapper (consistent encoding)")
        print(f"  Model parameters: {sum(p.numel() for p in _model.parameters()):,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to random moves.")


# Load model on import
_load_model()


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Thinking with Neural Network (raw policy)...")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Use raw model policy (MCTS was causing the bot to ignore model's good decisions!)
    if _model is not None and _move_mapper is not None:
        from .utils.board_encoder import board_to_tensor_torch
        
        # Get model's policy prediction
        board_tensor = board_to_tensor_torch(ctx.board)
        
        with torch.no_grad():
            policy_logits, value = _model(board_tensor)
        
        # Convert to probabilities
        policy_probs = torch.softmax(policy_logits, dim=1)[0]
        
        # Map to legal moves
        move_probs_list = []
        for move in legal_moves:
            idx = _move_mapper.get_move_index(move)
            prob = policy_probs[idx].item()
            move_probs_list.append((move, prob))
        
        # Normalize probabilities over legal moves only
        total_prob = sum(prob for _, prob in move_probs_list)
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs_list}
        else:
            # Uniform if no probability mass on legal moves
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}
        
        # Select best move
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        
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
