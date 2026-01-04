#!/bin/bash

# Optimized Training Script for MacBook Pro 2019 with AMD GPU
# Uses MPS (Metal Performance Shaders) for GPU acceleration

echo "🚀 Starting MobileNetV2-CBAM Training with AMD GPU Acceleration"
echo "=================================================="
echo ""

# Navigate to project directory
cd "/Users/macbookpro/Documents/Paper Implementation/mobilenetv2_food101"

# Activate virtual environment
source venv/bin/activate

# Check GPU availability
echo "📊 Checking GPU availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS (AMD GPU): {torch.backends.mps.is_available()}')"
echo ""

# Training configuration optimized for MacBook Pro 2019
echo "⚙️  Training Configuration:"
echo "  - Model: MobileNetV2-CBAM"
echo "  - Device: MPS (AMD GPU)"
echo "  - Epochs: 50"
echo "  - Batch Size: 64 (optimized for AMD GPU memory)"
echo "  - Workers: 4"
echo "  - Learning Rate: 0.001"
echo ""

# Start training with optimized settings for AMD GPU
echo "🎯 Starting training..."
echo "Expected time: ~10-14 hours with AMD GPU"
echo "=================================================="
echo ""

python train_improved.py \
    --epochs 50 \
    -j 4 \
    -b 64 \
    --lr 0.001 \
    --pretrained \
    FOOD101

echo ""
echo "=================================================="
echo "✅ Training complete!"
echo "Check model_best_cbam.pth.tar for best model"
echo "=================================================="
