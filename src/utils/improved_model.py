"""
Improved CNN-based chess model architecture.
This matches the architecture trained on Modal.
"""

import torch
import torch.nn as nn


class ImprovedChessModel(nn.Module):
    """
    Improved CNN-based chess model.
    Uses convolutional layers to preserve spatial information.
    Architecture inspired by AlphaZero but simplified.
    
    Input: 8x8x12 board representation (12 channels for piece types)
    Output: 
    - Move probabilities (policy head): 4096 values
    - Position evaluation (value head): 1 value between -1 and 1
    """
    
    def __init__(self, hidden_size=512):
        """
        Initialize the improved chess model.
        
        Args:
            hidden_size: Size of the fully connected hidden layers
        """
        super(ImprovedChessModel, self).__init__()
        
        # Initial convolutional block
        self.conv_input = nn.Sequential(
            nn.Conv2d(12, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            self._make_residual_block(128, 128) for _ in range(4)
        ])
        
        # Additional convolutional layers
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
        # 4096 is large enough to encode most move patterns
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4096),
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
        """
        Create a residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Sequential module representing the residual block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
            
        Returns:
            policy_logits: Raw logits for move probabilities, shape (batch, 4096)
            value: Position evaluation, shape (batch, 1)
        """
        # Initial convolution
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
    
    def predict(self, x):
        """
        Predict move probabilities and position value.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
            
        Returns:
            move_probs: Softmax probabilities for moves, shape (batch, 4096)
            value: Position evaluation, shape (batch, 1)
        """
        policy_logits, value = self.forward(x)
        move_probs = torch.softmax(policy_logits, dim=1)
        return move_probs, value

