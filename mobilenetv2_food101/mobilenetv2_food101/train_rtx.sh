#!/bin/bash
# RTX 3070 Training Script for Linux
# Run this after transferring files to your RTX machine

echo "========================================"
echo "RTX 3070 MobileNetV2-CBAM Training"
echo "========================================"
echo ""

# Check CUDA
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Start training with FP16 for maximum speed
echo "Starting training with Mixed Precision (FP16)..."
echo "Expected time: 3-4 hours"
echo ""

python3 train_rtx.py --epochs 50 -j 8 -b 256 --lr 0.001 --pretrained --amp FOOD101

echo ""
echo "========================================"
echo "Training Complete!"
echo "Check model_best_rtx.pth.tar"
echo "========================================"
