# GPU-Accelerated Training Guide for MacBook Pro 2019

## 🎯 Your Hardware

**MacBook Pro 2019 with AMD GPU**
- GPU: AMD Radeon Pro (dedicated)
- PyTorch uses **MPS (Metal Performance Shaders)** for AMD GPUs on macOS
- Expected speedup: **3-4x faster** than CPU training

---

## ✅ GPU Status

Your setup already supports MPS (AMD GPU acceleration):
- PyTorch 2.2.2 ✅
- MPS available ✅
- Training script configured for MPS ✅

---

## 🚀 Start GPU-Accelerated Training

### Option 1: Quick Start (Recommended)
```bash
cd "/Users/macbookpro/Documents/Paper Implementation/mobilenetv2_food101"
./train_gpu.sh
```

### Option 2: Manual Command
```bash
cd "/Users/macbookpro/Documents/Paper Implementation/mobilenetv2_food101"
source venv/bin/activate
python train_improved.py --epochs 50 -j 4 -b 64 --lr 0.001 --pretrained FOOD101
```

---

## ⚙️ Optimized Settings for Your AMD GPU

| Setting | Value | Reason |
|---------|-------|--------|
| Batch Size | 64 | Optimized for AMD GPU memory |
| Workers | 4 | Balanced for your CPU |
| Epochs | 50 | Full training for best accuracy |
| Device | MPS | Automatic AMD GPU detection |

---

## ⏱️ Expected Training Time

**With AMD GPU (MPS)**:
- **Per epoch**: ~12-15 minutes
- **Total (50 epochs)**: ~10-14 hours
- **Speedup**: 3-4x faster than CPU

**Comparison**:
- CPU only: ~48 hours
- AMD GPU (MPS): ~10-14 hours ✅
- NVIDIA GPU (CUDA): ~8-12 hours

---

## 📊 Monitoring Training

The script will show progress like this:

```
Epoch: [0][10/595]  Time 2.1  Data 0.1  
Loss 2.8145  Acc@1 52.67  Acc@5 78.34

Epoch: [10][595/595]  Time 1.8  Data 0.08  
Loss 1.2345  Acc@1 75.23  Acc@5 92.45

Test: [595/595]  Time 1.2  Loss 1.1234  
Acc@1 80.45  Acc@5 94.23
* Acc@1 80.450
```

---

## 💾 Checkpoints

Models saved automatically:
- `checkpoint_cbam_epoch{N}.pth.tar` - Each epoch
- `model_best_cbam.pth.tar` - Best accuracy

---

## 🎯 Expected Results

After 50 epochs with AMD GPU:
- **Accuracy**: 85-87%
- **Training time**: 10-14 hours
- **Model size**: ~16MB

---

## 🔧 Troubleshooting

### If GPU runs out of memory:
```bash
# Reduce batch size to 32
python train_improved.py --epochs 50 -j 4 -b 32 --lr 0.001 --pretrained FOOD101
```

### To check GPU usage during training:
```bash
# Open Activity Monitor
# Go to Window → GPU History
# You should see GPU usage spike during training
```

### If MPS is not working:
```bash
# Check MPS availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# If False, training will fall back to CPU (slower but still works)
```

---

## 📈 Progress Tracking

Create a simple log file:
```bash
# Run training with output logging
./train_gpu.sh 2>&1 | tee training_log.txt
```

This saves all output to `training_log.txt` for later review.

---

## ✅ Ready to Start!

Your AMD GPU is ready for accelerated training. Simply run:

```bash
cd "/Users/macbookpro/Documents/Paper Implementation/mobilenetv2_food101"
./train_gpu.sh
```

**Expected outcome**: 85-87% accuracy in ~10-14 hours! 🚀
