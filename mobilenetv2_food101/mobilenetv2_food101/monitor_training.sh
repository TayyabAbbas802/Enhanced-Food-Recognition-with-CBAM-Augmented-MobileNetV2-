#!/bin/bash

# Training Monitor Script
# Check training progress and GPU usage

echo "📊 MobileNetV2-CBAM Training Monitor"
echo "===================================="
echo ""

# Check if training is running
if pgrep -f "train_improved.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo "   PID: $(pgrep -f 'train_improved.py')"
else
    echo "❌ Training is NOT running"
    exit 1
fi

echo ""
echo "📈 Latest Training Output:"
echo "-----------------------------------"
tail -n 20 training_output.log
echo "-----------------------------------"
echo ""

echo "💾 Saved Checkpoints:"
ls -lh checkpoint_cbam_*.pth.tar 2>/dev/null | tail -n 5 || echo "   No checkpoints yet"
echo ""

if [ -f "model_best_cbam.pth.tar" ]; then
    echo "🏆 Best Model: model_best_cbam.pth.tar"
    ls -lh model_best_cbam.pth.tar
else
    echo "⏳ Best model not saved yet"
fi

echo ""
echo "📝 Full log: training_output.log"
echo "   Use: tail -f training_output.log (to watch live)"
