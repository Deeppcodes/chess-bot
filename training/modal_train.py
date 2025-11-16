"""
Modal training script for chess neural network.
Uses GPU for faster training with improved CNN architecture.

Usage:
    modal run modal_train.py

To download trained model:
    modal volume get chess-models chess_model_best.pth ./chess_model.pth
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("chess-training")

# Define the image with all dependencies
# Note: NumPy must be <2.0 for compatibility with PyTorch 2.1.0
# Include data files in the image if they exist
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy>=1.24.0,<2.0.0",  # NumPy 1.x for PyTorch compatibility
        "torch==2.1.0",
        "chess>=1.10.0",
        "requests>=2.31.0",      # Download Lichess database
        "zstandard>=0.22.0",     # Decompress .zst files
    ])
)

# Note: Data files will be uploaded at runtime via volume or downloaded on Modal
# We'll handle file upload in the main() function using Modal's file upload capabilities

# Define volume for persistent storage of models
volume = modal.Volume.from_name("chess-models", create_if_missing=True)

# Helper function to upload data file to volume
# Note: This requires the file to be uploaded via Modal CLI first, or we'll download on Modal
# For now, we'll simplify and let Modal download/filter the data


@app.function(
    image=image,
    gpu="T4",  # T4 GPU (cost-effective for this workload)
    timeout=21600,  # 6 hour timeout (increased for 100K games dataset)
    volumes={"/models": volume},
)
def train_model(
    num_games: int = 100000,  # 100K games = 2.9M positions
    epochs: int = 200,  # Increased to 200 with early stopping
    batch_size: int = 512,  # Larger batch for better gradient estimates
    lr: float = 0.0005,  # Lower LR for finer tuning with more data
    hidden_size: int = 512,
    val_split: float = 0.15,  # 15% validation
    data_file: str = None,  # Path to pre-processed .npz file
):
    """
    Train the chess model on Modal with GPU using real Lichess games.
    
    Can use pre-processed data file or download/filter on-the-fly.
    
    Args:
        num_games: Number of games (only used if data_file is None)
        epochs: Number of training epochs (with early stopping, patience=5)
        batch_size: Batch size for training
        lr: Learning rate
        hidden_size: Size of hidden layers
        val_split: Validation split ratio
        data_file: Path to pre-processed .npz file (if None, downloads/filters on Modal)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from chess import Board, Move, PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING
    import chess.pgn
    import time
    import random
    import requests
    import zstandard as zstd
    import io
    from datetime import datetime
    import copy
    import os
    
    print("=" * 70)
    print("Chess Neural Network Training on Modal")
    print("=" * 70)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== MODEL ARCHITECTURE ====================
    
    class ImprovedChessModel(nn.Module):
        """
        Improved CNN-based chess model.
        Uses convolutional layers to preserve spatial information.
        Architecture inspired by AlphaZero but simplified.
        """
        def __init__(self, hidden_size=512):
            super(ImprovedChessModel, self).__init__()
            
            # Initial convolutional block
            self.conv_input = nn.Sequential(
                nn.Conv2d(12, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            
            # Residual blocks
            self.res_blocks = nn.ModuleList([
                self._make_residual_block(128, 128) for _ in range(4)
            ])
            
            # Additional conv layers
            self.conv_layers = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            
            # Calculate flattened size: 256 channels * 8 * 8
            conv_output_size = 256 * 8 * 8
            
            # Shared fully connected layers
            self.fc_shared = nn.Sequential(
                nn.Linear(conv_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            
            # Policy head - outputs move probabilities
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 4096),  # Large output for move encoding
            )
            
            # Value head - outputs position evaluation
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Tanh()  # Output between -1 and 1
            )
        
        def _make_residual_block(self, in_channels, out_channels):
            """Create a residual block."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )
        
        def forward(self, x):
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, 12, 8, 8)
            
            Returns:
                policy_logits: Shape (batch, 4096)
                value: Shape (batch, 1)
            """
            # Initial conv
            out = self.conv_input(x)
            
            # Residual blocks with skip connections
            for res_block in self.res_blocks:
                identity = out
                out = res_block(out)
                out = out + identity  # Skip connection
                out = torch.relu(out)
            
            # Additional conv layers
            out = self.conv_layers(out)
            
            # Flatten
            batch_size = x.size(0)
            flat = out.reshape(batch_size, -1)
            
            # Shared layers
            shared = self.fc_shared(flat)
            
            # Policy and value heads
            policy_logits = self.policy_head(shared)
            value = self.value_head(shared)
            
            return policy_logits, value
    
    # ==================== DATA GENERATION ====================
    
    def board_to_tensor(board: Board) -> np.ndarray:
        """
        Convert a chess board to tensor representation.
        
        Returns:
            numpy array of shape (8, 8, 12)
        """
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        
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
                tensor[rank, file, channel] = 1.0
        
        return tensor
    
    def download_and_process_lichess_games(
        num_games: int = 10000,
        min_rating: int = 2000,
        game_types: list = ["blitz", "rapid"],
        skip_first_moves: int = 10,
        skip_last_moves: int = 5,
        sample_every_nth: int = 2
    ):
        """
        Download and process Lichess games with elite filters and smart sampling.
        
        Args:
            num_games: Target number of games to process
            min_rating: Minimum player rating (default: 2000)
            game_types: List of game types to include (default: ["blitz", "rapid"])
            skip_first_moves: Number of opening moves to skip (default: 10)
            skip_last_moves: Number of endgame moves to skip (default: 5)
            sample_every_nth: Sample every Nth move (default: 2)
        
        Returns:
            board_states: Array of shape (N, 8, 8, 12)
            moves: List of move UCI strings
            outcomes: Array of outcomes (N,)
        """
        print(f"\n📥 Downloading and processing Lichess games...")
        print(f"  Target games: {num_games}")
        print(f"  Min rating: {min_rating}+")
        print(f"  Game types: {', '.join(game_types)}")
        print(f"  Filters: Rated only, no draws")
        print(f"  Sampling: Skip first {skip_first_moves}, last {skip_last_moves}, every {sample_every_nth}nd move")
        
        all_board_states = []
        all_moves = []
        all_outcomes = []
        games_processed = 0
        games_filtered = 0
        filter_reasons = {
            'rating': 0,
            'game_type': 0,
            'draws': 0,
            'too_short': 0,
            'malformed': 0,
            'unknown_result': 0
        }
        
        # Try to download latest monthly database
        # Lichess databases are named: lichess_db_standard_rated_YYYY-MM.pgn.zst
        current_date = datetime.now()
        months_to_try = []
        
        # Try current month and previous 3 months
        for i in range(4):
            year_month = current_date.replace(day=1)
            if i > 0:
                # Go back i months
                if year_month.month > i:
                    year_month = year_month.replace(month=year_month.month - i)
                else:
                    year_month = year_month.replace(year=year_month.year - 1, month=12 + year_month.month - i)
            months_to_try.append(year_month.strftime("%Y-%m"))
        
        dctx = zstd.ZstdDecompressor()
        database_downloaded = False
        
        for month_str in months_to_try:
            if games_processed >= num_games:
                break
                
            url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month_str}.pgn.zst"
            print(f"\n  Trying database: {month_str}...")
            
            # Retry logic for downloads
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Download with streaming
                    response = requests.get(url, stream=True, timeout=60)
                    if response.status_code == 200:
                        print(f"  ✓ Downloading {month_str} database...")
                        database_downloaded = True
                        
                        # Stream and decompress
                        stream_reader = dctx.stream_reader(response.raw)
                        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                        
                        # Parse games directly from stream
                        pgn_buffer = ""
                        buffer_size_limit = 50 * 1024 * 1024  # 50MB buffer limit (increased)
                        total_read = 0
                        max_read = 2 * 1024 * 1024 * 1024  # Limit to ~2GB decompressed (increased)
                        
                        while games_processed < num_games and total_read < max_read:
                            # Read chunk (larger chunks = faster)
                            chunk = text_stream.read(65536)  # 64KB chunks (8x larger)
                            if not chunk:
                                break
                            
                            pgn_buffer += chunk
                            total_read += len(chunk)
                            
                            # Try to parse complete games from buffer
                            while games_processed < num_games:
                                try:
                                    # Find end of game (double newline)
                                    game_end = pgn_buffer.find("\n\n", pgn_buffer.find("[Event"))
                                    if game_end == -1:
                                        # No complete game yet, read more
                                        if len(pgn_buffer) > buffer_size_limit:
                                            # Buffer too large, skip to next potential game start
                                            next_start = pgn_buffer.find("[Event", 1000)
                                            if next_start > 0:
                                                pgn_buffer = pgn_buffer[next_start:]
                                            else:
                                                break
                                        break
                                    
                                    # Extract one game
                                    game_text = pgn_buffer[:game_end + 2]
                                    pgn_buffer = pgn_buffer[game_end + 2:]
                                    
                                    # Early filtering: Check headers before full parsing
                                    # This saves time by skipping move parsing for filtered games
                                    headers_only = game_text.split("\n\n")[0] if "\n\n" in game_text else game_text
                                    
                                    # Quick header checks before full parsing
                                    white_elo_header = ""
                                    black_elo_header = ""
                                    time_control_header = ""
                                    result_header = ""
                                    event_header = ""
                                    
                                    for line in headers_only.split("\n"):
                                        if line.startswith("[WhiteElo"):
                                            white_elo_header = line.split('"')[1] if '"' in line else ""
                                        elif line.startswith("[BlackElo"):
                                            black_elo_header = line.split('"')[1] if '"' in line else ""
                                        elif line.startswith("[TimeControl"):
                                            time_control_header = line.split('"')[1] if '"' in line else ""
                                        elif line.startswith("[Result"):
                                            result_header = line.split('"')[1] if '"' in line else ""
                                        elif line.startswith("[Event"):
                                            event_header = line.split('"')[1] if '"' in line else ""
                                    
                                    # Quick rating check before parsing
                                    try:
                                        white_rating_pre = int(white_elo_header) if white_elo_header and white_elo_header.isdigit() else 0
                                        black_rating_pre = int(black_elo_header) if black_elo_header and black_elo_header.isdigit() else 0
                                    except (ValueError, AttributeError):
                                        games_filtered += 1
                                        filter_reasons['malformed'] += 1
                                        continue
                                    
                                    # Skip if rating too low (before expensive parsing)
                                    if white_rating_pre < min_rating or black_rating_pre < min_rating:
                                        games_filtered += 1
                                        filter_reasons['rating'] += 1
                                        continue
                                    
                                    # Skip draws early
                                    if result_header == "1/2-1/2":
                                        games_filtered += 1
                                        filter_reasons['draws'] += 1
                                        continue
                                    
                                    # Quick game type check
                                    event_lower_pre = event_header.lower()
                                    time_control_pre = time_control_header
                                    initial_time_pre = 0
                                    if time_control_pre:
                                        try:
                                            if "+" in time_control_pre:
                                                parts = time_control_pre.split("+")
                                                initial_time_pre = int(parts[0]) if parts[0].isdigit() else 0
                                            elif time_control_pre.isdigit():
                                                initial_time_pre = int(time_control_pre)
                                        except (ValueError, AttributeError):
                                            initial_time_pre = 0
                                    
                                    game_type_match_pre = False
                                    if "blitz" in game_types:
                                        if ("blitz" in event_lower_pre or 
                                            (initial_time_pre > 0 and 180 <= initial_time_pre <= 600)):
                                            game_type_match_pre = True
                                    if "rapid" in game_types and not game_type_match_pre:
                                        if ("rapid" in event_lower_pre or 
                                            (initial_time_pre > 0 and 600 < initial_time_pre <= 1800)):
                                            game_type_match_pre = True
                                    
                                    if not game_type_match_pre and initial_time_pre == 0:
                                        if white_rating_pre >= 2200 or black_rating_pre >= 2200:
                                            game_type_match_pre = True
                                    
                                    if not game_type_match_pre:
                                        games_filtered += 1
                                        filter_reasons['game_type'] += 1
                                        continue
                                    
                                    # Now parse the full game (only if headers passed)
                                    pgn_io = io.StringIO(game_text)
                                    game = chess.pgn.read_game(pgn_io)
                                    
                                    if game is None:
                                        games_filtered += 1
                                        filter_reasons['malformed'] += 1
                                        continue
                                    
                                    # Filter games (re-check with parsed headers for safety)
                                    white_elo = game.headers.get("WhiteElo", "")
                                    black_elo = game.headers.get("BlackElo", "")
                                    time_control = game.headers.get("TimeControl", "")
                                    result = game.headers.get("Result", "*")
                                    event = game.headers.get("Event", "").lower()
                                    
                                    # Use parsed headers (should match header check, but verify)
                                    try:
                                        white_rating = int(white_elo) if white_elo and white_elo.isdigit() else white_rating_pre
                                        black_rating = int(black_elo) if black_elo and black_elo.isdigit() else black_rating_pre
                                    except (ValueError, AttributeError):
                                        white_rating = white_rating_pre
                                        black_rating = black_rating_pre
                                    
                                    # Final safety check (should already pass from header check)
                                    if white_rating < min_rating or black_rating < min_rating:
                                        games_filtered += 1
                                        filter_reasons['rating'] += 1
                                        continue
                                    
                                    # Use pre-checked game type (already validated in header check)
                                    game_type_match = game_type_match_pre
                                    
                                    if not game_type_match:
                                        games_filtered += 1
                                        filter_reasons['game_type'] += 1
                                        continue
                                    
                                    # Final draw check (should already be filtered)
                                    if result == "1/2-1/2":
                                        games_filtered += 1
                                        filter_reasons['draws'] += 1
                                        continue
                                    
                                    # Extract positions with smart sampling
                                    board = game.board()
                                    game_moves_list = list(game.mainline_moves())
                                    
                                    if len(game_moves_list) < skip_first_moves + skip_last_moves + 1:
                                        games_filtered += 1
                                        filter_reasons['too_short'] += 1
                                        continue
                                    
                                    # Determine outcome
                                    if result == "1-0":
                                        outcome = 1.0  # White won
                                    elif result == "0-1":
                                        outcome = -1.0  # Black won
                                    else:
                                        games_filtered += 1
                                        filter_reasons['unknown_result'] += 1
                                        continue
                                    
                                    # Extract positions with sampling
                                    positions_extracted = 0
                                    for move_idx, move in enumerate(game_moves_list):
                                        # Skip first N moves
                                        if move_idx < skip_first_moves:
                                            board.push(move)
                                            continue
                                        
                                        # Skip last N moves
                                        if move_idx >= len(game_moves_list) - skip_last_moves:
                                            break
                                        
                                        # Sample every Nth move
                                        if (move_idx - skip_first_moves) % sample_every_nth == 0:
                                            # Store position before move
                                            all_board_states.append(board_to_tensor(board))
                                            all_moves.append(move.uci())
                                            all_outcomes.append(outcome)
                                            positions_extracted += 1
                                        
                                        board.push(move)
                                    
                                    if positions_extracted > 0:
                                        games_processed += 1
                                        if games_processed % 500 == 0:  # More frequent updates
                                            print(f"  Processed: {games_processed}/{num_games} games, "
                                                  f"Filtered: {games_filtered}, "
                                                  f"Positions: {len(all_board_states):,}")
                                
                                except Exception as e:
                                    # Skip malformed games
                                    games_filtered += 1
                                    filter_reasons['malformed'] += 1
                                    continue
                            
                            if games_processed >= num_games:
                                break
                        
                        if games_processed >= num_games:
                            break
                    else:
                        if retry < max_retries - 1:
                            print(f"  ⚠️ Download failed (status {response.status_code}), retrying... ({retry + 1}/{max_retries})")
                            time.sleep(2)
                            continue
                        else:
                            raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    if retry < max_retries - 1:
                        print(f"  ⚠️ Download error: {e}, retrying... ({retry + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        print(f"  ⚠️ Failed to download {month_str} after {max_retries} attempts: {e}")
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"  ⚠️ Error processing {month_str}: {e}, retrying... ({retry + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        print(f"  ⚠️ Failed to process {month_str} after {max_retries} attempts: {e}")
                        break
        
        if not database_downloaded:
            print("\n❌ Failed to download any Lichess database!")
            print("Falling back to random game generation...")
            # Fallback to random games
            return generate_training_data_fallback(num_games)
        
        if len(all_board_states) == 0:
            print("\n⚠️ No games found matching criteria!")
            print("Falling back to random game generation...")
            return generate_training_data_fallback(num_games)
        
        board_states = np.array(all_board_states)
        outcomes = np.array(all_outcomes)
        
        print(f"\n✓ Processed {games_processed} games")
        print(f"  Filtered out: {games_filtered} games")
        print(f"  Filter breakdown:")
        for reason, count in filter_reasons.items():
            if count > 0:
                print(f"    - {reason}: {count}")
        print(f"  Total positions: {len(board_states):,}")
        if games_processed > 0:
            print(f"  Wins: {np.sum(outcomes > 0):,}")
            print(f"  Losses: {np.sum(outcomes < 0):,}")
            print(f"  Average positions per game: {len(board_states) / games_processed:.1f}")
        
        return board_states, all_moves, outcomes
    
    def generate_training_data_fallback(num_games: int):
        """Fallback to random games if Lichess download fails."""
        print(f"\n📊 Generating fallback training data from {num_games} random games...")
        
        all_board_states = []
        all_moves = []
        all_outcomes = []
        
        for game_idx in range(num_games):
            if (game_idx + 1) % 500 == 0:
                print(f"  Progress: {game_idx + 1}/{num_games} games...")
            
            board = Board()
            game_positions = []
            game_moves = []
            
            move_count = 0
            max_moves = random.randint(20, 150)
            
            while move_count < max_moves and not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                
                captures = [m for m in legal_moves if board.is_capture(m)]
                if captures and random.random() < 0.3:
                    move = random.choice(captures)
                else:
                    move = random.choice(legal_moves)
                
                game_positions.append(board_to_tensor(board))
                game_moves.append(move.uci())
                
                board.push(move)
                move_count += 1
            
            if board.is_checkmate():
                outcome = 1.0 if not board.turn else -1.0
            else:
                outcome = 0.0
            
            for state, move in zip(game_positions, game_moves):
                all_board_states.append(state)
                all_moves.append(move)
                all_outcomes.append(outcome)
        
        board_states = np.array(all_board_states)
        outcomes = np.array(all_outcomes)
        
        print(f"✓ Generated {len(board_states)} training positions")
        return board_states, all_moves, outcomes
    
    # ==================== DATASET ====================
    
    class ChessDataset(Dataset):
        """Dataset for chess training."""
        
        def __init__(self, board_states, moves, outcomes):
            self.board_tensors = torch.from_numpy(board_states).float()
            # Permute to (N, 12, 8, 8) format
            self.board_tensors = self.board_tensors.permute(0, 3, 1, 2)
            
            # Create policy targets using unified encoding
            self.policy_targets = []
            for move_str in moves:
                # Unified encoding: from_square * 64 + to_square (0-4095)
                move = Move.from_uci(move_str)
                target_idx = move.from_square * 64 + move.to_square
                self.policy_targets.append(target_idx)
            self.policy_targets = torch.tensor(self.policy_targets, dtype=torch.long)
            
            # Value targets
            self.outcomes = torch.from_numpy(outcomes).float().unsqueeze(1)
        
        def __len__(self):
            return len(self.board_tensors)
        
        def __getitem__(self, idx):
            return {
                'board': self.board_tensors[idx],
                'policy_target': self.policy_targets[idx],
                'value_target': self.outcomes[idx]
            }
    
    # ==================== TRAINING ====================
    
    def train_epoch(model, dataloader, optimizer, policy_criterion, value_criterion, device):
        """Train for one epoch."""
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            boards = batch['board'].to(device)
            policy_targets = batch['policy_target'].to(device)
            value_targets = batch['value_target'].to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(boards)
            
            policy_loss = policy_criterion(policy_logits, policy_targets)
            value_loss = value_criterion(value_pred, value_targets)
            loss = policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def validate(model, dataloader, policy_criterion, value_criterion, device):
        """Validate the model."""
        model.eval()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                boards = batch['board'].to(device)
                policy_targets = batch['policy_target'].to(device)
                value_targets = batch['value_target'].to(device)
                
                policy_logits, value_pred = model(boards)
                
                policy_loss = policy_criterion(policy_logits, policy_targets)
                value_loss = value_criterion(value_pred, value_targets)
                loss = policy_loss + value_loss
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    # ==================== MAIN TRAINING LOOP ====================
    
    start_time = time.time()
    
    # Load data from pre-processed file or download/filter on Modal
    # If data_file is provided, it should be a path in the Modal volume (e.g., /models/lichess_data.npz)
    if data_file:
        # Try the provided path first
        if not os.path.exists(data_file):
            # If not found, try common locations
            possible_paths = [
                data_file,
                f"/models/{os.path.basename(data_file)}",
                f"/models/lichess_data.npz",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    data_file = path
                    break
            else:
                print(f"⚠️ Pre-processed data file not found at: {data_file}")
                print(f"   Tried: {possible_paths}")
                print(f"   Falling back to download/filter on Modal...")
                data_file = None
    
    if data_file and os.path.exists(data_file):
        print(f"\n📂 Loading pre-processed data from: {data_file}")
        try:
            data = np.load(data_file, allow_pickle=True)
            board_states = data['board_states']
            move_indices = data['move_indices']
            outcomes = data['outcomes']
            stats = data.get('stats', {})
            
            # Convert move indices back to UCI strings for dataset
            moves = []
            for idx in move_indices:
                from_sq = idx // 64
                to_sq = idx % 64
                move = Move(from_sq, to_sq)
                moves.append(move.uci())
            
            print(f"✓ Loaded dataset:")
            print(f"  Samples: {len(board_states):,}")
            print(f"  Board shape: {board_states.shape}")
            print(f"  Wins: {np.sum(outcomes > 0):,}")
            print(f"  Losses: {np.sum(outcomes < 0):,}")
            if isinstance(stats, dict) and 'games_processed' in stats:
                print(f"  Games processed: {stats['games_processed']}")
                print(f"  Avg positions/game: {stats.get('avg_positions_per_game', 0):.1f}")
            
            # Verify data integrity
            assert len(board_states) == len(moves) == len(outcomes), "Data length mismatch!"
            assert len(board_states) > 0, "Empty dataset!"
            print(f"  ✓ Data verification passed")
            
        except Exception as e:
            print(f"❌ Failed to load pre-processed data: {e}")
            print(f"   Falling back to download/filter on Modal...")
            data_file = None
    
    if not data_file or not os.path.exists(data_file):
        print(f"\n📥 Downloading and processing Lichess data on Modal...")
        board_states, moves, outcomes = download_and_process_lichess_games(
            num_games=num_games,
            min_rating=2000,
            game_types=["blitz", "rapid"],
            skip_first_moves=10,
            skip_last_moves=5,
            sample_every_nth=2
        )
        
        # Verify we got data
        if len(board_states) == 0:
            raise RuntimeError("No data extracted! Check filtering logic.")
        print(f"✓ Extracted {len(board_states):,} positions from {num_games} games")
    
    # Try to load existing model for transfer learning (if available)
    existing_model_path = '/models/chess_model_best.pth'
    start_epoch = 0
    
    # Create dataset
    dataset = ChessDataset(board_states, moves, outcomes)
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📚 Dataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    
    # Create model (try to load existing for transfer learning)
    model = ImprovedChessModel(hidden_size=hidden_size)
    model = model.to(device)
    
    # Try to load existing model weights for transfer learning
    try:
        if os.path.exists(existing_model_path):
            checkpoint = torch.load(existing_model_path, map_location=device)
            if checkpoint.get('model_type') == 'ImprovedChessModel':
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"\n🔄 Transfer learning: Loaded model from epoch {start_epoch}")
                print(f"   Previous best val_loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
            else:
                print(f"\n⚠️  Existing model type mismatch, training from scratch")
        else:
            print(f"\n🆕 Training new model from scratch")
    except Exception as e:
        print(f"\n⚠️  Could not load existing model: {e}")
        print(f"   Training from scratch")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    # Training loop with early stopping
    print(f"\n🚀 Starting training for {epochs} epochs (with early stopping, patience=20)...")
    if start_epoch > 0:
        print(f"   Continuing from epoch {start_epoch} (transfer learning)")
    print("-" * 70)

    best_val_loss = float('inf')
    patience = 20  # Increased patience for larger dataset
    patience_counter = 0
    best_model_state = None
    best_epoch = start_epoch
    final_epoch = start_epoch
    
    # Initialize best_val_loss from checkpoint if available
    try:
        if os.path.exists(existing_model_path):
            checkpoint = torch.load(existing_model_path, map_location=device)
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"   Previous best validation loss: {best_val_loss:.4f}")
    except:
        pass
    
    for epoch in range(start_epoch, start_epoch + epochs):
        final_epoch = epoch
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, 
            policy_criterion, value_criterion, device
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, 
            policy_criterion, value_criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(P: {train_metrics['policy_loss']:.4f}, V: {train_metrics['value_loss']:.4f}) | "
              f"Val Loss: {val_metrics['total_loss']:.4f} "
              f"(P: {val_metrics['policy_loss']:.4f}, V: {val_metrics['value_loss']:.4f}) | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping logic
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['total_loss'],
                'model_type': 'ImprovedChessModel',
                'hidden_size': hidden_size,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, '/models/chess_model_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_metrics['total_loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹️  Early stopping triggered!")
                print(f"  No improvement for {patience} epochs.")
                print(f"  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
                
                # Restore best model weights
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print(f"  ✓ Restored best model weights from epoch {best_epoch + 1}")
                break
        
        scheduler.step()
    
    # Save final model (use best model if early stopping occurred)
    if best_model_state is not None and patience_counter >= patience:
        # Early stopping occurred, best model already saved
        print(f"\n  Using best model from epoch {best_epoch + 1} (early stopping)")
    else:
        # Training completed normally, save final model
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': final_epoch + 1,
            'val_loss': val_metrics['total_loss'],
            'model_type': 'ImprovedChessModel',
            'hidden_size': hidden_size,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        torch.save(final_checkpoint, '/models/chess_model_final.pth')
    
    # Commit volume to persist changes
    volume.commit()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("-" * 70)
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"Final validation loss: {val_metrics['total_loss']:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Epochs completed: {final_epoch + 1}/{epochs}")
    print(f"Average time per epoch: {total_time/(final_epoch + 1):.1f} seconds")
    print("=" * 70)
    print("\nTo download the trained model, run:")
    print("  modal volume get chess-models chess_model_best.pth ./models/chess_model.pth")
    print("=" * 70)
    
    return best_val_loss


@app.local_entrypoint()
def main(data_file: str = None):
    """
    Run training on Modal.
    
    Args:
        data_file: Optional path to pre-processed .npz file (local path, will be uploaded)
                   If None, will auto-detect data/lichess_data.npz or download/filter on Modal.
                   Examples: "data/lichess_data.npz" or None
    """
    print("🚀 Launching chess training job on Modal GPU...")
    print("Training on real Lichess games (2000+ rating, blitz/rapid, no draws)")
    
    # Auto-detect data file if not provided
    from pathlib import Path
    if not data_file:
        # Check for common data file locations
        possible_files = [
            "data/lichess_data.npz",
            "training/data/lichess_data.npz",
        ]
        for file_path in possible_files:
            if Path(file_path).exists():
                data_file = file_path
                print(f"✓ Auto-detected data file: {data_file}")
                break
    
    # Configure to use pre-processed data if available
    modal_data_path = None
    if data_file:
        # Check if file exists locally
        local_path = Path(data_file)
        if local_path.exists():
            print(f"✓ Found pre-processed data file: {data_file}")
            print(f"  File size: {local_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"  📤 Note: Upload this file to Modal volume before training:")
            print(f"     modal volume put chess-models {data_file} lichess_data.npz")
            print(f"  Then it will be available at /models/lichess_data.npz")
            print(f"  For now, setting to use /models/lichess_data.npz (upload it first)")
            modal_data_path = "/models/lichess_data.npz"
            print("  v1-test: Training with verified pre-processed data (30,012 positions)")
        else:
            print(f"⚠️  Pre-processed data file not found: {data_file}")
            print(f"   Will download/filter games on Modal instead...")
            modal_data_path = None
    else:
        print("  v1: 5K games, 40 epochs - Scaled up to beat 1320 Stockfish")
        print("  Transfer learning: Continuing from v0 model")
    
    print("This will take approximately 20-25 minutes.\n")
    
    # Run training
    best_loss = train_model.remote(
        num_games=100000 if not modal_data_path else 1000,  # 100K games = 2.9M positions
        epochs=200,        # 200 epochs with early stopping (patience=20)
        batch_size=512,    # Larger batch for 2.9M positions
        lr=0.0005,         # Lower LR for finer tuning
        hidden_size=512,   # Hidden layer size
        val_split=0.15,    # 15% validation split
        data_file=modal_data_path,  # Pre-processed data file path in Modal container
    )
    
    print(f"\n✅ Training job completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print("\nNext steps:")
    print("1. Download the model:")
    print("   modal volume get chess-models chess_model_best.pth ./models/chess_model.pth")
    print("2. Benchmark the model:")
    print("   python benchmarks/benchmark_stockfish.py --progressive")
    print("3. The model will be ready to use with your chess bot!")

