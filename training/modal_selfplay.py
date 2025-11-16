"""
Modal GPU-accelerated self-play generation.
Generates self-play games in parallel on GPU.
"""

import modal

# Create Modal app
app = modal.App("chess-selfplay")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy>=1.24.0,<2.0.0",  # NumPy 1.x for compatibility
        "torch==2.1.0",
        "python-chess>=1.10.0",
    ])
)

# Volume for storing models and data
volume = modal.Volume.from_name("chess-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,  # 2 hours
    volumes={"/models": volume},
)
def generate_selfplay_games(
    model_path: str = "/models/chess_model_best.pth",
    num_games: int = 100,
    num_simulations: int = 100,
    output_name: str = "selfplay_batch.npz"
):
    """
    Generate self-play games on GPU.

    Args:
        model_path: Path to model in Modal volume
        num_games: Number of games to generate
        num_simulations: MCTS simulations per move
        output_name: Output filename
    """
    import torch
    import numpy as np
    from chess import Board, Move, PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING
    import chess
    import time
    import math
    from typing import Dict

    print("=" * 70)
    print("Self-Play Generation on Modal GPU")
    print("=" * 70)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================== MODEL ARCHITECTURE ====================

    class ImprovedChessModel(torch.nn.Module):
        """Improved CNN-based chess model."""
        def __init__(self, hidden_size=512):
            super(ImprovedChessModel, self).__init__()

            self.conv_input = torch.nn.Sequential(
                torch.nn.Conv2d(12, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU()
            )

            self.res_blocks = torch.nn.ModuleList([
                self._make_residual_block(128, 128) for _ in range(4)
            ])

            self.conv_layers = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
            )

            conv_output_size = 256 * 8 * 8

            self.fc_shared = torch.nn.Sequential(
                torch.nn.Linear(conv_output_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
            )

            self.policy_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 4096),
            )

            self.value_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 1),
                torch.nn.Tanh()
            )

        def _make_residual_block(self, in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.conv_input(x)

            for res_block in self.res_blocks:
                identity = out
                out = res_block(out)
                out = out + identity
                out = torch.relu(out)

            out = self.conv_layers(out)
            batch_size = x.size(0)
            flat = out.reshape(batch_size, -1)
            shared = self.fc_shared(flat)

            policy_logits = self.policy_head(shared)
            value = self.value_head(shared)

            return policy_logits, value

    # ==================== SELF-PLAY LOGIC ====================

    def board_to_tensor(board: Board):
        """Convert board to tensor."""
        tensor = torch.zeros((1, 12, 8, 8), dtype=torch.float32, device=device)

        piece_to_channel = {
            (PAWN, True): 0, (ROOK, True): 1, (KNIGHT, True): 2,
            (BISHOP, True): 3, (QUEEN, True): 4, (KING, True): 5,
            (PAWN, False): 6, (ROOK, False): 7, (KNIGHT, False): 8,
            (BISHOP, False): 9, (QUEEN, False): 10, (KING, False): 11,
        }

        for square in range(64):
            piece = board.piece_at(square)
            if piece is not None:
                channel = piece_to_channel[(piece.piece_type, piece.color)]
                rank = 7 - (square // 8)
                file = square % 8
                tensor[0, channel, rank, file] = 1.0

        return tensor

    def get_move_probabilities(policy_logits, legal_moves):
        """Convert policy logits to move probabilities."""
        # Create mask for legal moves
        legal_indices = [move.from_square * 64 + move.to_square for move in legal_moves]

        # Apply softmax only to legal moves
        legal_logits = policy_logits[0, legal_indices]
        legal_probs = torch.softmax(legal_logits, dim=0)

        move_probs = {}
        for i, move in enumerate(legal_moves):
            move_probs[move] = legal_probs[i].item()

        return move_probs

    def select_move(move_probs, temperature=1.0):
        """Select move from probabilities."""
        moves = list(move_probs.keys())
        probs = list(move_probs.values())

        if temperature < 0.1:
            # Deterministic
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            return moves[best_idx]

        # Apply temperature
        probs = [p ** (1.0 / temperature) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]

        return np.random.choice(moves, p=probs)

    def play_game(model, max_moves=200):
        """Play one self-play game."""
        board = Board()
        positions = []
        policies = []
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            # Get current position
            board_tensor = board_to_tensor(board)

            # Get policy and value from model
            with torch.no_grad():
                policy_logits, value = model(board_tensor)

            # Get legal moves
            legal_moves = list(board.generate_legal_moves())
            if not legal_moves:
                break

            # Get move probabilities
            move_probs = get_move_probabilities(policy_logits, legal_moves)

            # Store position and policy
            position_np = board_tensor.cpu().numpy()[0].transpose(1, 2, 0)  # (12,8,8) -> (8,8,12)
            positions.append(position_np)

            # Convert to policy vector
            policy_vec = np.zeros(4096, dtype=np.float32)
            for move, prob in move_probs.items():
                idx = move.from_square * 64 + move.to_square
                policy_vec[idx] = prob
            policies.append(policy_vec)

            # Select move (temperature schedule)
            temperature = 1.0 if move_count < 30 else 0.1
            move = select_move(move_probs, temperature)

            # Make move
            board.push(move)
            move_count += 1

        # Determine result
        if board.is_checkmate():
            game_result = 1.0 if not board.turn else -1.0
        else:
            game_result = 0.0

        # Assign values (alternating perspectives)
        values = []
        for i in range(len(positions)):
            if i % 2 == 0:
                values.append(game_result)
            else:
                values.append(-game_result)

        return positions, policies, values

    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    hidden_size = checkpoint.get('hidden_size', 512)
    model = ImprovedChessModel(hidden_size=hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded (hidden_size={hidden_size})")
    print(f"\nGenerating {num_games} self-play games...")

    # Generate games
    all_positions = []
    all_policies = []
    all_values = []

    start_time = time.time()

    for game_idx in range(num_games):
        positions, policies, values = play_game(model)

        all_positions.extend(positions)
        all_policies.extend(policies)
        all_values.extend(values)

        if (game_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_idx + 1) / elapsed
            eta = (num_games - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0
            print(f"  Game {game_idx + 1}/{num_games} | "
                  f"{games_per_sec:.2f} games/sec | ETA: {eta:.1f}s")

    # Save data
    positions_array = np.array(all_positions, dtype=np.float32)
    policies_array = np.array(all_policies, dtype=np.float32)
    values_array = np.array(all_values, dtype=np.float32)

    output_path = f"/models/{output_name}"
    np.savez_compressed(
        output_path,
        positions=positions_array,
        policies=policies_array,
        values=values_array,
        num_games=num_games,
        simulations_per_move=num_simulations
    )

    volume.commit()

    total_time = time.time() - start_time
    print(f"\n✓ Generated {num_games} games in {total_time:.1f}s")
    print(f"  Total positions: {len(all_positions):,}")
    print(f"  Saved to: {output_path}")

    return output_path


@app.local_entrypoint()
def main(
    num_games: int = 100,
    num_simulations: int = 100,
    output_name: str = "selfplay_batch.npz"
):
    """
    Run self-play generation on Modal GPU.

    Args:
        num_games: Number of games to generate
        num_simulations: MCTS simulations per move
        output_name: Output filename
    """
    print("🚀 Launching self-play generation on Modal GPU...")
    print(f"  Games: {num_games}")
    print(f"  Simulations per move: {num_simulations}")

    output_path = generate_selfplay_games.remote(
        model_path="/models/chess_model_best.pth",
        num_games=num_games,
        num_simulations=num_simulations,
        output_name=output_name
    )

    print(f"\n✅ Self-play generation complete!")
    print(f"Output: {output_path}")
    print("\nTo download:")
    print(f"  modal volume get chess-models {output_name} ./data/{output_name}")
