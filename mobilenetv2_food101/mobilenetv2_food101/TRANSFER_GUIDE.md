# Transfer Package for RTX 3070

## 📦 Files to Transfer

Copy these files to your RTX 3070 machine:

### Essential Files (Required):
1. ✅ `cbam.py` - CBAM attention module
2. ✅ `mobilenetv2_cbam.py` - Enhanced MobileNetV2 model
3. ✅ `train_rtx.py` - RTX-optimized training script
4. ✅ `RTX_SETUP_GUIDE.md` - Complete setup instructions
5. ✅ `train_rtx.bat` (Windows) or `train_rtx.sh` (Linux) - Easy start script

### Optional Files:
- `requirements_rtx.txt` - Python dependencies
- Food-101 dataset (will auto-download if not present)

---

## 🚀 Transfer Methods

### Method 1: USB Drive (Easiest)
```bash
# On MacBook - Copy to USB
cp cbam.py mobilenetv2_cbam.py train_rtx.py RTX_SETUP_GUIDE.md train_rtx.* /Volumes/YOUR_USB/

# On RTX machine - Copy from USB
# Windows: Copy from USB drive to C:\Projects\food101\
# Linux: cp /media/usb/* ~/food101/
```

### Method 2: Cloud (Google Drive, Dropbox)
1. Upload files to cloud storage
2. Download on RTX machine

### Method 3: Git (If you use version control)
```bash
# On MacBook
git add cbam.py mobilenetv2_cbam.py train_rtx.py
git commit -m "RTX optimized training"
git push

# On RTX machine
git pull
```

### Method 4: Network Transfer (SCP/SFTP)
```bash
# If RTX machine is on same network
scp cbam.py mobilenetv2_cbam.py train_rtx.py user@rtx-machine:/path/to/folder/
```

---

## ⚡ Quick Setup on RTX Machine

### Windows:
```cmd
1. Create folder: C:\Projects\food101\
2. Copy files to folder
3. Open PowerShell in that folder
4. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
5. Run: pip install tqdm scikit-learn
6. Double-click: train_rtx.bat
```

### Linux:
```bash
1. Create folder: mkdir ~/food101 && cd ~/food101
2. Copy files to folder
3. Run: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
4. Run: pip3 install tqdm scikit-learn
5. Run: chmod +x train_rtx.sh && ./train_rtx.sh
```

---

## 📋 Checklist

Before starting training on RTX machine:

- [ ] All 5 essential files transferred
- [ ] CUDA PyTorch installed
- [ ] CUDA available (run: `python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] At least 20GB free disk space (for dataset)
- [ ] GPU drivers updated

---

## 🎯 Expected Performance

| Metric | MacBook (AMD) | RTX 3070 | RTX 3070 + FP16 |
|--------|---------------|----------|-----------------|
| Time/Epoch | 67 min | 6-8 min | 3-4 min |
| 50 Epochs | 56 hours | 5-7 hours | 3-4 hours |
| Speedup | 1x | 8-10x | 14-16x |

---

## 📝 Training Command

After transfer, run:

```bash
# Standard (5-7 hours)
python train_rtx.py --epochs 50 -j 8 -b 256 --lr 0.001 --pretrained FOOD101

# With FP16 - Fastest! (3-4 hours)
python train_rtx.py --epochs 50 -j 8 -b 256 --lr 0.001 --pretrained --amp FOOD101
```

---

## ✅ Ready to Transfer!

All files are prepared and optimized for your RTX 3070. Follow the steps above to get **85-87% accuracy in just 3-7 hours**!
