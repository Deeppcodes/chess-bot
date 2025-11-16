"""
Phase 2: Training on self-play data (Reinforcement Learning).
Improves model by training on its own games.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import copy


class SelfPlayDataset(Dataset):
    """Dataset for self-play training data."""

    def __init__(self, positions, policies, values):
        """
        Initialize dataset.

        Args:
            positions: Board states (N, 8, 8, 12)
            policies: Policy targets (N, 4096)
            values: Value targets (N,)
        """
        self.positions = torch.from_numpy(positions).float()
        # Permute to (N, 12, 8, 8) format
        self.positions = self.positions.permute(0, 3, 1, 2)

        self.policies = torch.from_numpy(policies).float()
        self.values = torch.from_numpy(values).float().unsqueeze(1)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }


def train_epoch(model, dataloader, optimizer, policy_criterion, value_criterion, device):
    """Train for one epoch."""
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        positions = batch['position'].to(device)
        policy_targets = batch['policy'].to(device)
        value_targets = batch['value'].to(device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred = model(positions)

        # Compute losses
        # Policy loss: cross entropy with soft targets
        policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_logits, dim=1)) / policy_targets.size(0)

        # Value loss: MSE
        value_loss = value_criterion(value_pred, value_targets)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward pass
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
            positions = batch['position'].to(device)
            policy_targets = batch['policy'].to(device)
            value_targets = batch['value'].to(device)

            policy_logits, value_pred = model(positions)

            policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_logits, dim=1)) / policy_targets.size(0)
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


def train_on_selfplay(
    model_path: str,
    selfplay_data_path: str,
    output_path: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.0001,
    val_split: float = 0.1,
    device: str = 'cpu'
):
    """
    Train model on self-play data (Phase 2).

    Args:
        model_path: Path to Phase 1 model
        selfplay_data_path: Path to self-play data .npz
        output_path: Path to save Phase 2 model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split ratio
        device: Device to train on
    """
    print("=" * 70)
    print("Phase 2: Training on Self-Play Data")
    print("=" * 70)

    # Load Phase 1 model
    from src.utils.improved_model import ImprovedChessModel

    print(f"\nLoading Phase 1 model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint.get('model_type', 'ImprovedChessModel')
    hidden_size = checkpoint.get('hidden_size', 512)

    model = ImprovedChessModel(hidden_size=hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"  Model loaded: {model_type}, hidden_size={hidden_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load self-play data
    print(f"\nLoading self-play data from: {selfplay_data_path}")
    data = np.load(selfplay_data_path)

    positions = data['positions']
    policies = data['policies']
    values = data['values']

    print(f"  Loaded {len(positions):,} positions")
    print(f"  Games: {data.get('num_games', 'unknown')}")
    print(f"  Simulations per move: {data.get('simulations_per_move', 'unknown')}")

    # Create dataset
    dataset = SelfPlayDataset(positions, policies, values)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Training: {train_size:,}")
    print(f"  Validation: {val_size:,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Loss functions and optimizer
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    # Training loop
    print(f"\nTraining for {epochs} epochs (batch_size={batch_size}, lr={lr})")
    print("-" * 70)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    policy_criterion, value_criterion, device)

        # Validate
        val_metrics = validate(model, val_loader,
                              policy_criterion, value_criterion, device)

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(P: {train_metrics['policy_loss']:.4f}, V: {train_metrics['value_loss']:.4f}) | "
              f"Val Loss: {val_metrics['total_loss']:.4f} "
              f"(P: {val_metrics['policy_loss']:.4f}, V: {val_metrics['value_loss']:.4f}) | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())

            # Save best model
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['total_loss'],
                'model_type': 'ImprovedChessModel',
                'hidden_size': hidden_size,
                'phase': 2,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, output_path)
            print(f"  ✓ Saved best model (val_loss: {val_metrics['total_loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹️  Early stopping triggered!")
                print(f"  No improvement for {patience} epochs.")
                print(f"  Best validation loss: {best_val_loss:.4f}")

                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print(f"  ✓ Restored best model weights")
                break

        scheduler.step()

    print("\n" + "=" * 70)
    print("✅ Phase 2 Training Complete!")
    print("-" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train on self-play data (Phase 2)")
    parser.add_argument('--model', type=str, default='chess_model_best.pth',
                       help='Path to Phase 1 model')
    parser.add_argument('--selfplay-data', type=str, default='data/selfplay.npz',
                       help='Path to self-play data')
    parser.add_argument('--output', type=str, default='models/chess_model_phase2.pth',
                       help='Output path for Phase 2 model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    train_on_selfplay(
        model_path=args.model,
        selfplay_data_path=args.selfplay_data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
