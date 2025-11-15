"""
Neural network model for chess move prediction and position evaluation.
Uses a CNN/ResNet architecture for stronger performance with MCTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and skip connection.
    This helps the network learn deeper representations by allowing gradients
    to flow through skip connections.
    """
    
    def __init__(self, channels=64):
        """
        Initialize residual block.
        
        Args:
            channels: Number of input/output channels (kept constant)
        """
        super(ResidualBlock, self).__init__()
        
        # First convolution: 3x3 conv, same padding to keep spatial size
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolution: 3x3 conv, same padding
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, channels, 8, 8)
            
        Returns:
            Output tensor of same shape as input
        """
        # Save input for skip connection
        residual = x
        
        # First conv block: Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block: Conv -> BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection and apply ReLU
        out = out + residual  # Skip connection
        out = F.relu(out)
        
        return out


class ChessModel(nn.Module):
    """
    CNN/ResNet model for chess move prediction and position evaluation.
    
    Architecture:
    - Input: (batch, 12, 8, 8) board representation
    - Initial 3x3 convolution to 64 channels
    - 4-8 residual blocks for feature extraction
    - Two separate heads: policy (move probabilities) and value (position evaluation)
    
    This architecture preserves spatial information and learns local patterns
    better than an MLP, leading to stronger MCTS performance.
    """
    
    def __init__(self, num_residual_blocks=6, channels=64, dropout=0.0, use_torchscript=False):
        """
        Initialize the chess CNN/ResNet model.
        
        Args:
            num_residual_blocks: Number of residual blocks (4-8 recommended, default: 6)
            channels: Number of channels in convolutional layers (default: 64)
            dropout: Dropout probability for regularization (0.0 = disabled, default: 0.0)
            use_torchscript: Whether to prepare for TorchScript export (default: False)
        """
        super(ChessModel, self).__init__()
        
        self.channels = channels
        self.use_torchscript = use_torchscript
        
        # Initial convolution: 12 input channels -> 64 channels
        # 3x3 kernel with padding=1 keeps spatial dimensions (8x8)
        self.initial_conv = nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(channels)
        
        # Residual blocks: stack multiple residual blocks for deeper learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None
        
        # POLICY HEAD: Predicts move probabilities
        # 1x1 conv reduces channels, then flatten and fully connected to 256 move indices
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        # Flatten: 32 * 8 * 8 = 2048
        self.policy_fc = nn.Linear(32 * 8 * 8, 256)  # 256 move indices (same as before)
        
        # VALUE HEAD: Evaluates position strength
        # 1x1 conv reduces channels, then fully connected layers to single value
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        # Flatten: 32 * 8 * 8 = 2048
        self.value_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        # Tanh activation ensures output is between -1 and 1
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
            
        Returns:
            policy_logits: Raw logits for move probabilities, shape (batch, 256)
            value: Position evaluation, shape (batch, 1)
        """
        # Initial convolution: (batch, 12, 8, 8) -> (batch, 64, 8, 8)
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Optional dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # POLICY HEAD: Extract move probabilities
        policy = self.policy_conv(x)  # (batch, 32, 8, 8)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten: (batch, 2048)
        policy_logits = self.policy_fc(policy)  # (batch, 256)
        
        # VALUE HEAD: Evaluate position
        value = self.value_conv(x)  # (batch, 32, 8, 8)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten: (batch, 2048)
        value = self.value_fc1(value)  # (batch, 64)
        value = F.relu(value)
        value = self.value_fc2(value)  # (batch, 1)
        value = torch.tanh(value)  # Output between -1 and 1
        
        return policy_logits, value
    
    def predict(self, x):
        """
        Predict move probabilities and position value.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
            
        Returns:
            move_probs: Softmax probabilities for moves, shape (batch, 256)
            value: Position evaluation, shape (batch, 1)
        """
        policy_logits, value = self.forward(x)
        move_probs = F.softmax(policy_logits, dim=1)
        return move_probs, value
    
    def export_torchscript(self, example_input=None):
        """
        Export model to TorchScript for faster inference during MCTS.
        
        Args:
            example_input: Example input tensor (batch, 12, 8, 8). 
                          If None, creates a dummy tensor.
        
        Returns:
            TorchScript model
        """
        if example_input is None:
            example_input = torch.randn(1, 12, 8, 8)
        
        self.eval()
        traced_model = torch.jit.trace(self, example_input)
        return traced_model

