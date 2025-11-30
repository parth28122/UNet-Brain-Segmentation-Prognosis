"""
Attention U-Net Implementation for Brain Tumor Segmentation

Based on: "Attention U-Net: Learning Where to Look for the Pancreas"
Oktay et al., 2018 - https://arxiv.org/abs/1804.03999

This implementation follows the paper's architecture with:
- Attention Gates between encoder-decoder skip connections
- Additive attention formulation
- Grid-attention mechanism (not global vector attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionGate(nn.Module):
    """
    Attention Gate Module (Grid Attention)
    
    Implements the attention mechanism from the paper:
    - Gating signal g from coarser scale
    - Feature map x from skip connection
    - Attention coefficients αᵢ = sigmoid(qᵢ)
    - Linear transformations using 1×1 convolutions
    
    Args:
        F_g: Number of channels in gating signal
        F_l: Number of channels in feature map
        F_int: Number of intermediate channels
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        # Linear transformations (implemented as 1×1 convs)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention gate.
        
        Args:
            g: Gating signal from coarser scale [B, F_g, H_g, W_g]
            x: Feature map from skip connection [B, F_l, H_l, W_l]
        
        Returns:
            Attention-weighted feature map [B, F_l, H_l, W_l]
        """
        # Ensure spatial dimensions match (upsample gating signal if needed)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample gating signal to match feature map size
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # Additive attention (Equation 1 in paper)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention coefficients (Equation 2 in paper)
        return x * psi


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super(DoubleConv, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block with attention gate."""
    
    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        use_attention: bool = True
    ):
        super(Up, self).__init__()
        
        self.use_attention = use_attention
        self.bilinear = bilinear
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            up_channels = decoder_channels
        else:
            self.up = nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2)
            up_channels = decoder_channels // 2
        
        if use_attention:
            self.attention = AttentionGate(
                F_g=up_channels,
                F_l=skip_channels,
                F_int=max(skip_channels // 2, 1)
            )
        
        self.conv = DoubleConv(up_channels + skip_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x1: Feature map from decoder path (coarser scale)
            x2: Feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle dimension mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Apply attention gate if enabled
        if self.use_attention:
            x2 = self.attention(g=x1, x=x2)
        
        # Concatenate and convolve
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Brain Tumor Segmentation.
    
    Architecture:
    - Encoder path: Downsampling with double convolutions
    - Decoder path: Upsampling with attention gates
    - Skip connections: Connected through attention gates
    
    Args:
        in_channels: Number of input channels (e.g., 4 for BRATS: T1, T1ce, T2, FLAIR)
        num_classes: Number of output classes (1 for binary segmentation)
        base_filters: Number of base filters (default: 64)
        depth: Depth of the network (default: 4)
        use_attention: Whether to use attention gates (default: True)
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        use_attention: bool = True,
        dropout: float = 0.2
    ):
        super(AttentionUNet, self).__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        
        # Encoder path
        self.inc = DoubleConv(in_channels, base_filters)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        encoder_channels = [base_filters]
        filters = base_filters
        for _ in range(depth):
            out_channels = filters * 2
            self.down_blocks.append(Down(filters, out_channels))
            filters = out_channels
            encoder_channels.append(filters)
        
        # Bottleneck
        self.bottleneck = DoubleConv(filters, filters)
        decoder_channels = filters
        
        # Decoder path with attention
        self.up_blocks = nn.ModuleList()
        skip_channels = encoder_channels[::-1][:depth]
        for skip_ch in skip_channels:
            out_ch = decoder_channels // 2
            self.up_blocks.append(
                Up(
                    decoder_channels=decoder_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    bilinear=True,
                    use_attention=use_attention
                )
            )
            decoder_channels = out_ch
        
        # Output layer
        self.final_channels = decoder_channels
        self.outc = nn.Sequential(
            nn.Conv2d(self.final_channels, self.final_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.final_channels // 2, num_classes, kernel_size=1),
            nn.Sigmoid()  # For binary segmentation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Segmentation mask [B, num_classes, H, W]
        """
        # Encoder path
        x1 = self.inc(x)
        skip_connections = [x1]
        
        x = x1
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with attention
        skip_connections = skip_connections[::-1][:len(self.up_blocks)]
        for up_block, skip in zip(self.up_blocks, skip_connections):
            x = up_block(x, skip)
        
        # Output
        logits = self.outc(x)
        return logits


def get_model(config: dict) -> AttentionUNet:
    """
    Factory function to create Attention U-Net model from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        AttentionUNet model instance
    """
    model_config = config.get("model", {})
    return AttentionUNet(
        in_channels=model_config.get("in_channels", 4),
        num_classes=model_config.get("num_classes", 1),
        base_filters=model_config.get("base_filters", 64),
        depth=model_config.get("depth", 4),
        use_attention=model_config.get("attention", True),
        dropout=model_config.get("dropout", 0.2)
    )


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=4, num_classes=1, base_filters=64, depth=4)
    model = model.to(device)
    
    # Test input
    x = torch.randn(2, 4, 256, 256).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


