"""
Training script for the chess neural network model.
Trains on generated training data with policy and value losses.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from chess import Move
import os

from src.utils.model import ChessModel
from src.utils.board_encoder import board_to_tensor_torch
from src.utils.data_collection import load_training_data
from src.utils.move_mapper import MoveMapper


class ChessDataset(Dataset):
    """Dataset for chess training data."""
    
    def __init__(self, board_states, moves, outcomes, move_mapper):
        """
        Initialize dataset.
        
        Args:
            board_states: Array of shape (N, 8, 8, 12)
            moves: List of move UCI strings
            outcomes: Array of outcomes (N,)
            move_mapper: MoveMapper instance
        """
        self.board_states = board_states
        self.moves = moves
        self.outcomes = outcomes
        self.move_mapper = move_mapper
        
        # Convert board states to torch tensors
        self.board_tensors = torch.from_numpy(board_states).float()
        # Permute to (N, 12, 8, 8) format
        self.board_tensors = self.board_tensors.permute(0, 3, 1, 2)
        
        # Create policy targets (class indices for CrossEntropyLoss)
        self.policy_targets = []
        for move_str in moves:
            move = Move.from_uci(move_str)
            target_idx = move_mapper.get_move_index(move)
            self.policy_targets.append(target_idx)
        self.policy_targets = torch.tensor(self.policy_targets, dtype=torch.long)
        
        # Convert outcomes to torch tensors
        self.outcomes = torch.from_numpy(outcomes).float().unsqueeze(1)  # (N, 1)
    
    def __len__(self):
        return len(self.board_states)
    
    def __getitem__(self, idx):
        return {
            'board': self.board_tensors[idx],
            'policy_target': self.policy_targets[idx],
            'value_target': self.outcomes[idx]
        }


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
        
        # Forward pass
        policy_logits, value_pred = model(boards)
        
        # Compute losses
        policy_loss = policy_criterion(policy_logits, policy_targets)
        value_loss = value_criterion(value_pred, value_targets)
        
        # Combined loss (weighted)
        loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
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
            
            # Forward pass
            policy_logits, value_pred = model(boards)
            
            # Compute losses
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


def main():
    parser = argparse.ArgumentParser(description='Train chess neural network model')
    parser.add_argument('--data', type=str, default='training_data.npz',
                       help='Training data file (default: training_data.npz)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden layer size (default: 256)')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of hidden layers (default: 3)')
    parser.add_argument('--output', type=str, default='chess_model.pth',
                       help='Output model file (default: chess_model.pth)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 3: Model Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data
    print(f"\nLoading training data from {args.data}...")
    board_states, moves, outcomes = load_training_data(args.data)
    print(f"Loaded {len(board_states)} training positions")
    
    # Create move mapper
    move_mapper = MoveMapper()
    
    # Create dataset
    dataset = ChessDataset(board_states, moves, outcomes, move_mapper)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = ChessModel(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers)
    model = model.to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, 
                                   policy_criterion, value_criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, 
                               policy_criterion, value_criterion, device)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Policy: {train_metrics['policy_loss']:.4f}, "
              f"Value: {train_metrics['value_loss']:.4f}, "
              f"Total: {train_metrics['total_loss']:.4f}")
        print(f"  Val   - Policy: {val_metrics['policy_loss']:.4f}, "
              f"Value: {val_metrics['value_loss']:.4f}, "
              f"Total: {val_metrics['total_loss']:.4f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'move_mapper': move_mapper,
                'epoch': epoch,
                'val_loss': val_metrics['total_loss'],
            }, args.output)
            print(f"  ✓ Saved best model to {args.output}")
        print()
    
    print("=" * 60)
    print(f"✓ Training complete! Best model saved to {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

