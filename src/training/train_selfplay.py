"""
Training script for self-play generated data with policy and value targets.
Handles new format: (boards, policy_targets, value_targets)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import time
import os
import glob

from src.utils.model import ChessModel


class SelfPlayDataset(Dataset):
    """Dataset for self-play training data with policy and value targets."""
    
    def __init__(self, boards, policy_targets, value_targets):
        """
        Args:
            boards: (N, 12, 8, 8) board tensors
            policy_targets: (N, 256) policy distributions
            value_targets: (N,) game outcomes
        """
        self.boards = torch.from_numpy(boards).float()
        self.policy_targets = torch.from_numpy(policy_targets).float()
        self.value_targets = torch.from_numpy(value_targets).float()
    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        return self.boards[idx], self.policy_targets[idx], self.value_targets[idx]


def train_epoch(model, train_loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0

    use_amp = scaler is not None

    for boards, policy_targets, value_targets in train_loader:
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(enabled=use_amp):
            policy_logits, value_pred = model(boards)

            # Policy loss: cross-entropy with soft targets
            policy_loss = -(policy_targets * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

            # Value loss: MSE
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets)

            # Combined loss
            loss = policy_loss + value_loss

        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

    num_batches = len(train_loader)
    return {
        'total': total_loss / num_batches,
        'policy': policy_loss_sum / num_batches,
        'value': value_loss_sum / num_batches
    }


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    
    with torch.no_grad():
        for boards, policy_targets, value_targets in val_loader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            policy_logits, value_pred = model(boards)
            
            policy_loss = -(policy_targets * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets)
            
            loss = policy_loss + value_loss
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
    
    num_batches = len(val_loader)
    return {
        'total': total_loss / num_batches,
        'policy': policy_loss_sum / num_batches,
        'value': value_loss_sum / num_batches
    }


def load_multiple_datasets(data_paths):
    """Load and merge multiple NPZ files."""
    all_boards = []
    all_policies = []
    all_values = []

    for path in data_paths:
        print(f"  Loading {path}...")
        data = np.load(path)

        # Handle different key names
        boards = data.get('boards', data.get('board_states'))
        policies = data.get('policies', data.get('policy_targets'))
        values = data.get('values', data.get('value_targets'))

        all_boards.append(boards)
        all_policies.append(policies)
        all_values.append(values)
        print(f"    → {len(boards)} positions")

    # Concatenate all datasets
    boards = np.concatenate(all_boards, axis=0)
    policies = np.concatenate(all_policies, axis=0)
    values = np.concatenate(all_values, axis=0)

    return boards, policies, values


def main():
    parser = argparse.ArgumentParser(description='Train on self-play data')
    parser.add_argument('--data', type=str, default=None,
                       help='Single training data file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing multiple .npz files')
    parser.add_argument('--model', type=str, default=None,
                       help='Base model to fine-tune (e.g., chess_model_best.pth)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Maximum epochs (default: 5 for Phase 2)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split')
    parser.add_argument('--num-residual-blocks', type=int, default=6,
                       help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=64,
                       help='Number of channels')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--output', type=str, default='chess_model_sp_v1.pth',
                       help='Output model file (default: chess_model_sp_v1.pth)')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision (AMP) for faster training')

    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PHASE 2: SELF-PLAY TRAINING - AlphaZero Style")
    print("="*70)
    print(f"✓ Dropout: {args.dropout}")
    print(f"✓ Learning Rate Scheduling: Enabled")
    print(f"✓ Early Stopping: {args.patience} epochs")
    print(f"✓ Gradient Clipping: Enabled")
    print(f"✓ Mixed Precision (AMP): {'Enabled' if args.use_amp else 'Disabled'}")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data - support both single file and directory
    if args.data_dir:
        print(f"\nLoading training data from directory: {args.data_dir}")
        data_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
        if not data_files:
            raise ValueError(f"No .npz files found in {args.data_dir}")
        print(f"Found {len(data_files)} dataset files")
        boards, policy_targets, value_targets = load_multiple_datasets(data_files)
    elif args.data:
        print(f"\nLoading training data from {args.data}...")
        data = np.load(args.data)
        boards = data.get('boards', data.get('board_states'))
        policy_targets = data.get('policies', data.get('policy_targets'))
        value_targets = data.get('values', data.get('value_targets'))
    else:
        raise ValueError("Must specify either --data or --data-dir")

    print(f"Total positions loaded: {len(boards)}")

    # Create dataset
    dataset = SelfPlayDataset(boards, policy_targets, value_targets)

    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {train_size}, Val: {val_size}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create or load model
    if args.model:
        print(f"\nLoading base model from {args.model}...")
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model = ChessModel(
            num_residual_blocks=checkpoint.get('num_residual_blocks', args.num_residual_blocks),
            channels=checkpoint.get('channels', args.channels),
            dropout=args.dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        start_epoch = 0  # Start fresh for Phase 2
        best_val_loss = float('inf')
    else:
        print(f"\nCreating new model from scratch...")
        model = ChessModel(
            num_residual_blocks=args.num_residual_blocks,
            channels=args.channels,
            dropout=args.dropout
        ).to(device)
        start_epoch = 0
        best_val_loss = float('inf')

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")
    print(f"Architecture: {args.num_residual_blocks} blocks, {args.channels} channels")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # AMP Scaler
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    if scaler:
        print(f"✓ Using AMP for faster training")

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training for up to {args.epochs} epochs...")
    print(f"{'='*70}\n")

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, scaler)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update LR
        scheduler.step(val_metrics['total'])

        epoch_time = time.time() - epoch_start

        # Print
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Policy: {train_metrics['policy']:.4f}, "
              f"Value: {train_metrics['value']:.4f}, Total: {train_metrics['total']:.4f}")
        print(f"  Val   - Policy: {val_metrics['policy']:.4f}, "
              f"Value: {val_metrics['value']:.4f}, Total: {val_metrics['total']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': train_metrics['total'],
                'model_type': 'ChessModel',
                'num_residual_blocks': args.num_residual_blocks,
                'channels': args.channels,
                'dropout': args.dropout
            }, args.output)

            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n{'='*70}")
                print(f"⚠ Early stopping after {epoch+1} epochs")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"{'='*70}")
                break

        print()

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"  Best model: {args.output}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


