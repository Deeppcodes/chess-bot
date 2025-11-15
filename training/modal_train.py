"""
Modal training script for chess neural network.
Uses GPU for faster training with improved CNN architecture.

Usage:
    modal run modal_train.py

To download trained model:
    modal volume get chess-models chess_model_best.pth ./chess_model.pth
"""

import modal

# Create Modal app
app = modal.App("chess-training")

# Define the image with all dependencies
# Note: NumPy must be <2.0 for compatibility with PyTorch 2.1.0
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy>=1.24.0,<2.0.0",  # NumPy 1.x for PyTorch compatibility
        "torch==2.1.0",
        "chess>=1.10.0",
    ])
)

# Define volume for persistent storage of models
volume = modal.Volume.from_name("chess-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU (good price/performance) or "A100" for faster training
    timeout=7200,  # 2 hour timeout
    volumes={"/models": volume},
)
def train_model(
    num_games: int = 5000,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.001,
    hidden_size: int = 512,
    val_split: float = 0.2,
):
    """
    Train the chess model on Modal with GPU.
    
    Args:
        num_games: Number of games to generate for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        hidden_size: Size of hidden layers
        val_split: Validation split ratio
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from chess import Board, Move, PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING
    import time
    import random
    
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
    
    def generate_training_data(num_games: int):
        """
        Generate training data from random games.
        
        Returns:
            board_states: Array of shape (N, 8, 8, 12)
            moves: List of move UCI strings
            outcomes: Array of outcomes (N,)
        """
        print(f"\n📊 Generating training data from {num_games} games...")
        
        all_board_states = []
        all_moves = []
        all_outcomes = []
        
        for game_idx in range(num_games):
            if (game_idx + 1) % 500 == 0:
                print(f"  Progress: {game_idx + 1}/{num_games} games...")
            
            board = Board()
            game_positions = []
            game_moves = []
            
            # Play random game with some basic heuristics
            move_count = 0
            max_moves = random.randint(20, 150)
            
            while move_count < max_moves and not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                
                # Add some intelligence: prefer captures occasionally
                captures = [m for m in legal_moves if board.is_capture(m)]
                if captures and random.random() < 0.3:
                    move = random.choice(captures)
                else:
                    move = random.choice(legal_moves)
                
                # Store position and move
                game_positions.append(board_to_tensor(board))
                game_moves.append(move.uci())
                
                board.push(move)
                move_count += 1
            
            # Determine outcome
            if board.is_checkmate():
                # Winner is the player who just moved
                outcome = 1.0 if not board.turn else -1.0
            else:
                outcome = 0.0  # Draw
            
            # Add to dataset
            for state, move in zip(game_positions, game_moves):
                all_board_states.append(state)
                all_moves.append(move)
                all_outcomes.append(outcome)
        
        board_states = np.array(all_board_states)
        outcomes = np.array(all_outcomes)
        
        print(f"✓ Generated {len(board_states)} training positions")
        print(f"  Wins: {np.sum(outcomes > 0)}")
        print(f"  Draws: {np.sum(outcomes == 0)}")
        print(f"  Losses: {np.sum(outcomes < 0)}")
        
        return board_states, all_moves, outcomes
    
    # ==================== DATASET ====================
    
    class ChessDataset(Dataset):
        """Dataset for chess training."""
        
        def __init__(self, board_states, moves, outcomes):
            self.board_tensors = torch.from_numpy(board_states).float()
            # Permute to (N, 12, 8, 8) format
            self.board_tensors = self.board_tensors.permute(0, 3, 1, 2)
            
            # Create simple policy targets (move hash for now)
            self.policy_targets = []
            for move_str in moves:
                # Simple hash: use from_square * 64 + to_square
                move = Move.from_uci(move_str)
                target_idx = move.from_square * 64 + move.to_square
                target_idx = target_idx % 4096  # Keep within bounds
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
    
    # Generate data
    board_states, moves, outcomes = generate_training_data(num_games)
    
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
    
    # Create model
    model = ImprovedChessModel(hidden_size=hidden_size)
    model = model.to(device)
    
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
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
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
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
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
        
        scheduler.step()
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
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
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {val_metrics['total_loss']:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average time per epoch: {total_time/epochs:.1f} seconds")
    print("=" * 70)
    print("\nTo download the trained model, run:")
    print("  modal volume get chess-models chess_model_best.pth ./chess_model.pth")
    print("=" * 70)
    
    return best_val_loss


@app.local_entrypoint()
def main():
    """Run training on Modal."""
    print("🚀 Launching chess training job on Modal GPU...")
    print("This will take approximately 10-30 minutes depending on settings.\n")
    
    # Run training
    best_loss = train_model.remote(
        num_games=5000,    # Generate 5000 games
        epochs=30,         # Train for 30 epochs
        batch_size=256,    # Batch size
        lr=0.001,          # Learning rate
        hidden_size=512,   # Hidden layer size
        val_split=0.2,     # 20% validation split
    )
    
    print(f"\n✅ Training job completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print("\nNext steps:")
    print("1. Download the model:")
    print("   modal volume get chess-models chess_model_best.pth ./models/chess_model.pth")
    print("2. The model will be ready to use with your chess bot!")

