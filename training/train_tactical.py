"""
Phase 3: Training on tactical puzzles.
Improves tactical awareness by training on curated puzzles.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import copy


class TacticalDataset(Dataset):
    """Dataset for tactical puzzle training."""

    def __init__(self, positions, move_indices, values):
        """
        Initialize dataset.

        Args:
            positions: Board states (N, 8, 8, 12)
            move_indices: Move targets (N,) - indices 0-4095
            values: Value targets (N,)
        """
        self.positions = torch.from_numpy(positions).float()
        # Permute to (N, 12, 8, 8) format
        self.positions = self.positions.permute(0, 3, 1, 2)

        self.move_indices = torch.from_numpy(move_indices).long()
        self.values = torch.from_numpy(values).float().unsqueeze(1)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'move': self.move_indices[idx],
            'value': self.values[idx]
        }


def train_epoch(model, dataloader, optimizer, policy_criterion, value_criterion, device,
                policy_weight: float = 2.0):
    """Train for one epoch with emphasis on policy (tactical moves)."""
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    correct_moves = 0
    total_moves = 0

    for batch in dataloader:
        positions = batch['position'].to(device)
        move_targets = batch['move'].to(device)
        value_targets = batch['value'].to(device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred = model(positions)

        # Policy loss (cross entropy)
        policy_loss = policy_criterion(policy_logits, move_targets)

        # Value loss (MSE)
        value_loss = value_criterion(value_pred, value_targets)

        # Combined loss with higher weight on policy for tactics
        loss = policy_weight * policy_loss + value_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(policy_logits, 1)
        correct_moves += (predicted == move_targets).sum().item()
        total_moves += move_targets.size(0)

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
        num_batches += 1

    accuracy = correct_moves / total_moves if total_moves > 0 else 0

    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': total_loss / num_batches,
        'accuracy': accuracy
    }


def validate(model, dataloader, policy_criterion, value_criterion, device,
            policy_weight: float = 2.0):
    """Validate the model."""
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    correct_moves = 0
    total_moves = 0

    with torch.no_grad():
        for batch in dataloader:
            positions = batch['position'].to(device)
            move_targets = batch['move'].to(device)
            value_targets = batch['value'].to(device)

            policy_logits, value_pred = model(positions)

            policy_loss = policy_criterion(policy_logits, move_targets)
            value_loss = value_criterion(value_pred, value_targets)
            loss = policy_weight * policy_loss + value_loss

            # Calculate accuracy
            _, predicted = torch.max(policy_logits, 1)
            correct_moves += (predicted == move_targets).sum().item()
            total_moves += move_targets.size(0)

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

    accuracy = correct_moves / total_moves if total_moves > 0 else 0

    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': total_loss / num_batches,
        'accuracy': accuracy
    }


def train_on_tactics(
    model_path: str,
    tactical_data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 0.0001,
    policy_weight: float = 2.0,
    val_split: float = 0.1,
    device: str = 'cpu'
):
    """
    Train model on tactical puzzles (Phase 3).

    Args:
        model_path: Path to Phase 2 model (or Phase 1 if skipping Phase 2)
        tactical_data_path: Path to tactical puzzle data .npz
        output_path: Path to save Phase 3 model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        policy_weight: Weight for policy loss (higher = more emphasis on tactics)
        val_split: Validation split ratio
        device: Device to train on
    """
    print("=" * 70)
    print("Phase 3: Training on Tactical Puzzles")
    print("=" * 70)

    # Load model
    from src.utils.improved_model import ImprovedChessModel

    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint.get('model_type', 'ImprovedChessModel')
    hidden_size = checkpoint.get('hidden_size', 512)
    phase = checkpoint.get('phase', 1)

    model = ImprovedChessModel(hidden_size=hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"  Model loaded: {model_type}, hidden_size={hidden_size}, phase={phase}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tactical data
    print(f"\nLoading tactical data from: {tactical_data_path}")
    data = np.load(tactical_data_path, allow_pickle=True)

    positions = data['positions']
    move_indices = data['move_indices']
    values = data['values']

    print(f"  Loaded {len(positions):,} tactical puzzles")
    print(f"  Puzzles: {data.get('num_puzzles', 'unknown')}")
    print(f"  Rating range: {data.get('rating_range', 'unknown')}")

    # Create dataset
    dataset = TacticalDataset(positions, move_indices, values)

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
    print(f"\nTraining for {epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Policy weight: {policy_weight}x (emphasis on tactical accuracy)")
    print("-" * 70)

    best_val_accuracy = 0.0
    patience = 10
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    policy_criterion, value_criterion, device,
                                    policy_weight)

        # Validate
        val_metrics = validate(model, val_loader,
                              policy_criterion, value_criterion, device,
                              policy_weight)

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['accuracy']*100:.1f}% | "
              f"Val: Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['accuracy']*100:.1f}% | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping based on accuracy
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())

            # Save best model
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_metrics['total_loss'],
                'model_type': 'ImprovedChessModel',
                'hidden_size': hidden_size,
                'phase': 3,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, output_path)
            print(f"  ✓ Saved best model (accuracy: {val_metrics['accuracy']*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹️  Early stopping triggered!")
                print(f"  No improvement for {patience} epochs.")
                print(f"  Best validation accuracy: {best_val_accuracy*100:.1f}%")

                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print(f"  ✓ Restored best model weights")
                break

        scheduler.step()

    print("\n" + "=" * 70)
    print("✅ Phase 3 Training Complete!")
    print("-" * 70)
    print(f"Best validation accuracy: {best_val_accuracy*100:.1f}%")
    print(f"Model saved to: {output_path}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train on tactical puzzles (Phase 3)")
    parser.add_argument('--model', type=str, default='models/chess_model_phase2.pth',
                       help='Path to Phase 2 model')
    parser.add_argument('--tactical-data', type=str, default='data/tactical_puzzles.npz',
                       help='Path to tactical puzzle data')
    parser.add_argument('--output', type=str, default='models/chess_model_phase3.pth',
                       help='Output path for Phase 3 model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--policy-weight', type=float, default=2.0,
                       help='Weight for policy loss')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    train_on_tactics(
        model_path=args.model,
        tactical_data_path=args.tactical_data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        policy_weight=args.policy_weight,
        device=args.device
    )
