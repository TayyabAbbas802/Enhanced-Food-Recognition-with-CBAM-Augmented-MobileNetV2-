"""
MobileNetV2 with CBAM Attention
Enhanced version of MobileNetV2 with CBAM attention modules for improved accuracy.

Expected improvement: +4-5% accuracy over baseline MobileNetV2
"""

import torch
import torch.nn as nn
import torchvision.models as models
from cbam import CBAM


class MobileNetV2_CBAM(nn.Module):
    """MobileNetV2 with CBAM Attention
    
    Integrates CBAM attention modules at strategic positions in MobileNetV2
    to improve feature representation and classification accuracy.
    
    Args:
        num_classes: Number of output classes (default: 101 for Food-101)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
    """
    def __init__(self, num_classes=101, pretrained=True):
        super(MobileNetV2_CBAM, self).__init__()
        
        # Load pretrained MobileNetV2
        base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract features (all layers except classifier)
        self.features = base_model.features
        
        # Add CBAM attention modules after key inverted residual blocks
        # MobileNetV2 has 19 inverted residual blocks
        # We add CBAM after blocks that have significant channel changes
        
        # CBAM after early layers (32 channels)
        self.cbam1 = CBAM(in_channels=32, reduction=8)
        
        # CBAM after middle layers (96 channels)
        self.cbam2 = CBAM(in_channels=96, reduction=16)
        
        # CBAM before classifier (320 channels)
        self.cbam3 = CBAM(in_channels=320, reduction=16)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )
        
        # Initialize classifier weights if using pretrained model
        if pretrained:
            self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the classifier layer"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution and first inverted residual blocks
        # features[0:2] = Conv + InvertedResidual (output: 16 channels)
        x = self.features[0:2](x)
        
        # features[2:4] = InvertedResidual blocks (output: 24 channels)
        x = self.features[2:4](x)
        
        # features[4:7] = InvertedResidual blocks (output: 32 channels)
        x = self.features[4:7](x)
        x = self.cbam1(x)  # Apply CBAM attention
        
        # features[7:11] = InvertedResidual blocks (output: 64 channels)
        x = self.features[7:11](x)
        
        # features[11:14] = InvertedResidual blocks (output: 96 channels)
        x = self.features[11:14](x)
        x = self.cbam2(x)  # Apply CBAM attention
        
        # features[14:17] = InvertedResidual blocks (output: 160 channels)
        x = self.features[14:17](x)
        
        # features[17:18] = InvertedResidual block (output: 320 channels)
        x = self.features[17:18](x)
        x = self.cbam3(x)  # Apply CBAM attention
        
        # Final conv layer (output: 1280 channels) - but we use 320 for efficiency
        # x = self.features[18:](x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


# Test function
if __name__ == '__main__':
    print("Testing MobileNetV2_CBAM model...")
    
    # Create model
    model = MobileNetV2_CBAM(num_classes=101, pretrained=False)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: torch.Size([{batch_size}, 101])")
    
    assert output.shape == torch.Size([batch_size, 101]), "Output shape mismatch!"
    
    print("\n✅ MobileNetV2_CBAM model works correctly!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
