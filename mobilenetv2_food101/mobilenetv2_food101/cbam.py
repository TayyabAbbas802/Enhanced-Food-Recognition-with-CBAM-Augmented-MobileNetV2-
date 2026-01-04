"""
CBAM (Convolutional Block Attention Module) Implementation
Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)

This module implements channel and spatial attention mechanisms
to improve feature representation in CNNs.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module
    
    Focuses on 'what' is meaningful in the input feature map.
    Uses both average and max pooling to aggregate spatial information.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module
    
    Focuses on 'where' is meaningful in the input feature map.
    Uses channel pooling to generate spatial attention map.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    
    Combines channel and spatial attention sequentially.
    Can be inserted after any convolutional layer.
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for channel attention (default: 16)
        kernel_size: Kernel size for spatial attention (default: 7)
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Test function
if __name__ == '__main__':
    # Test CBAM module
    print("Testing CBAM module...")
    
    # Create dummy input
    batch_size = 4
    channels = 64
    height, width = 56, 56
    x = torch.randn(batch_size, channels, height, width)
    
    # Create CBAM module
    cbam = CBAM(in_channels=channels)
    
    # Forward pass
    output = cbam(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ CBAM module works correctly!")
    
    # Test with different channel sizes
    for ch in [32, 96, 320]:
        cbam_test = CBAM(in_channels=ch)
        x_test = torch.randn(2, ch, 28, 28)
        out_test = cbam_test(x_test)
        assert out_test.shape == x_test.shape, f"Shape mismatch for {ch} channels"
        print(f"✅ CBAM works for {ch} channels")
    
    print("\n✅ All tests passed!")
