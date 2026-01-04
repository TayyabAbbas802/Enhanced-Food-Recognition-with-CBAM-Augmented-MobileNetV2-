# MobileNetV2 Food-101 Project - Quick Start

## 🎯 Project Overview

This is a working implementation of MobileNetV2 for Food-101 classification with **76.3% baseline accuracy**.

**Repository**: Cloned from https://github.com/AlexKoff88/mobilenetv2_food101

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start training (dataset auto-downloads)
python train.py --epochs 30 -j 6 -b 256 --lr 0.001 --pretrained --wd 10e-5 FOOD101
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install torch torchvision

# 3. Run training
python train.py --epochs 30 -j 6 -b 256 --lr 0.001 --pretrained --wd 10e-5 FOOD101
```

---

## 📊 Expected Results

- **Baseline Accuracy**: 76.3% (Top-1) on Food-101
- **Training Time**: ~8 hours (GPU) / ~48 hours (CPU)
- **Dataset Size**: ~5GB (auto-downloads on first run)
- **Model Size**: ~14MB

---

## 📁 Project Structure

```
mobilenetv2_food101/
├── train.py              # Main training script
├── validate_checkpoint.py # Validation script
├── export.py             # ONNX export
├── requirements.txt      # Dependencies
├── setup.sh             # Setup script (NEW)
├── checkpoints/         # Model checkpoints
└── FOOD101/             # Dataset (auto-created)
```

---

## 🎓 Training Details

### Key Features:
- ✅ Pre-trained MobileNetV2 (ImageNet weights)
- ✅ Automatic Food-101 dataset download
- ✅ Data augmentation (RandomResizedCrop, RandomHorizontalFlip)
- ✅ Learning rate scheduling (StepLR)
- ✅ Checkpoint saving (best model)
- ✅ MPS support (Apple Silicon)

### Training Command Breakdown:
```bash
python train.py \
  --epochs 30          # Number of epochs
  -j 6                 # Number of workers
  -b 256               # Batch size
  --lr 0.001           # Learning rate
  --pretrained         # Use ImageNet pre-trained weights
  --wd 10e-5           # Weight decay
  FOOD101              # Dataset directory
```

---

## 📈 Monitoring Training

The training script will output:
- Epoch progress
- Training loss and accuracy (Top-1, Top-5)
- Validation accuracy
- Best model checkpoint

Example output:
```
Epoch: [0][10/298]  Time 2.345 (2.456)  Data 0.123 (0.145)  
Loss 3.2145e+00 (3.4567e+00)  Acc@1 45.67 (42.34)  Acc@5 72.34 (69.12)
```

---

## 💾 Model Checkpoints

Checkpoints are saved automatically:
- `checkpoint.pth.tar` - Latest checkpoint
- `model_best.pth.tar` - Best performing model

### Load Pre-trained Model:
```bash
# Download from HuggingFace
# https://huggingface.co/AlexKoff88/mobilenet_v2_food101

# Validate checkpoint
python validate_checkpoint.py model_best.pth.tar
```

---

## 🔧 Hardware Requirements

### Minimum:
- CPU: Multi-core processor
- RAM: 8GB
- Storage: 10GB free space

### Recommended:
- GPU: NVIDIA GPU with 8GB+ VRAM (CUDA)
- Apple Silicon: M1/M2/M3 (MPS support)
- RAM: 16GB+
- Storage: 20GB free space

---

## 🎯 Next Steps: Improvements

After reproducing the baseline, implement these improvements:

### Week 2: Add CBAM Attention
- Target: 80-81% accuracy (+4-5%)
- See: `../BEST_EXISTING_IMPLEMENTATION.md`

### Week 3: Advanced Augmentation
- AutoAugment, CutMix, MixUp
- Target: 83-85% accuracy

### Week 4: Hyperparameter Tuning
- CosineAnnealing scheduler
- Longer training (50 epochs)
- Target: 85-87% accuracy

---

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size
```bash
python train.py --epochs 30 -j 6 -b 128 --lr 0.001 --pretrained --wd 10e-5 FOOD101
```

### Issue: Slow Training
**Solution**: Reduce workers or use GPU
```bash
python train.py --epochs 30 -j 2 -b 256 --lr 0.001 --pretrained --wd 10e-5 FOOD101
```

### Issue: Dataset Download Fails
**Solution**: The script auto-downloads. If it fails, check internet connection and retry.

---

## 📊 Baseline Results (Expected)

After 30 epochs:
- **Top-1 Accuracy**: ~76.3%
- **Top-5 Accuracy**: ~93.5%
- **Training Loss**: ~0.8
- **Validation Loss**: ~1.2

---

## 🔗 Resources

- **Original Repository**: https://github.com/AlexKoff88/mobilenetv2_food101
- **Pre-trained Model**: https://huggingface.co/AlexKoff88/mobilenet_v2_food101
- **Food-101 Dataset**: http://data.vision.ee.ethz.ch/cvl/food-101.html
- **Improvement Guide**: See `../BEST_EXISTING_IMPLEMENTATION.md`

---

## ✅ Verification Checklist

- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Training started successfully
- [ ] Dataset downloaded (5GB)
- [ ] First epoch completed
- [ ] Checkpoints being saved
- [ ] Baseline accuracy achieved (~76%)

---

## 🎉 Success!

Once you achieve 76.3% accuracy, you've successfully reproduced the baseline!

Next: Follow the improvement roadmap in `../BEST_EXISTING_IMPLEMENTATION.md` to reach 85-87% accuracy.
