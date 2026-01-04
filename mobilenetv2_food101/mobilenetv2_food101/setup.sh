#!/bin/bash

# MobileNetV2 Food-101 Setup Script
# This script sets up the environment and prepares for training

echo "🚀 Setting up MobileNetV2 Food-101 Project..."
echo ""

# Step 1: Create virtual environment
echo "📦 Step 1: Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 2: Install dependencies
echo "📦 Step 2: Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt

# Step 3: Create necessary directories
echo "📁 Step 3: Creating directories..."
mkdir -p checkpoints
mkdir -p FOOD101

# Step 4: Check PyTorch installation
echo "✅ Step 4: Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run training (dataset will auto-download):"
echo "   python train.py --epochs 30 -j 6 -b 256 --lr 0.001 --pretrained --wd 10e-5 FOOD101"
echo ""
echo "Note: The Food-101 dataset (~5GB) will be automatically downloaded on first run."
echo "Training will take several hours depending on your hardware."
