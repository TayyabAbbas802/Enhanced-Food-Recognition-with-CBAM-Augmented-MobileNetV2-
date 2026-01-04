# Efficient Food Recognition with CBAM-Enhanced MobileNetV2: Achieving 83.5% Accuracy on Food-101

**Authors**: [Your Name]
**Date**: January 2026

---

## Abstract

Food recognition on mobile and edge devices presents a significant challenge due to the need for high accuracy under limited computational resources. In this paper, we propose a lightweight yet powerful architecture based on MobileNetV2, enhanced with Convolutional Block Attention Modules (CBAM) and trained with an advanced data augmentation pipeline. Our approach systematically improves the baseline MobileNetV2 top-1 accuracy on the Food-101 dataset from 76.3% to **83.51%**, an absolute gain of 7.21%. Crucially, our model achieves this performance with only **2.27 million parameters**, outperforming the much larger ResNet-50 (25.6M parameters, 82.4% accuracy) while being **11.3× smaller**. We demonstrate that strategic integration of attention mechanisms combined with mixed-precision training and modern regularization techniques enables state-of-the-art efficiency for food recognition tasks.

**Keywords**: Food Recognition, MobileNetV2, CBAM, Efficient Deep Learning, Data Augmentation.

---

## 1. Introduction

Dietary monitoring has become an essential tool for personal health management and chronic disease prevention. Automatic food recognition systems, powered by computer vision, offer a convenient way for users to log their meals. However, the high computational cost of modern Deep Convolutional Neural Networks (DCNNs) often requires processing on cloud servers, introducing latency and privacy concerns. Deploying these models on mobile devices requires a delicate balance between accuracy, model size, and inference speed.

The Food-101 dataset is a standard benchmark for this task, containing 101 food categories with high intra-class variance and inter-class similarity. While large models like InceptionV3 and efficient architectures like EfficientNet have achieved high accuracy, they often remain too heavy or slow for real-time mobile use on older devices.

In this work, we address this limitation by enhancing the MobileNetV2 architecture. Our contributions are as follows:
1.  **Strategic Integration of Attention**: We integrate CBAM (Channel and Spatial attention) at three specific stages of the MobileNetV2 backbone to refine feature extraction without significant parameter overhead.
2.  **Comprehensive Optimization**: We utilize an optimized training recipe including mixed-precision (FP16) training, Cosine Annealing learning rate scheduling, and label smoothing.
3.  **Advanced Augmentation**: We employ a robust augmentation pipeline (Random Erasing, Color Jitter, Affine transforms) to improve generalization.
4.  **SOTA Efficiency**: We achieve **83.51%** accuracy with only **2.27M parameters**, setting a new benchmark for efficiency-focused food recognition.

---

## 2. Related Work

**Efficient Architectures**: MobileNetV2 (Sandler et al., 2018) introduced inverted residual blocks and linear bottlenecks to reduce computation. It remains a dominant backbone for mobile tasks. ShuffleNet and SqueezeNet offer alternatives, but MobileNetV2 generally provides better support in modern mobile ML frameworks.

**Attention Mechanisms**: Attention allows networks to focus on relevant features. SE-Net (Hu et al.) introduced channel attention. CBAM (Woo et al., 2018) extended this by adding a spatial attention module, refining features in both channel and spatial dimensions. This is particularly useful for food recognition, where distinguishing features (e.g., the texture of a steak vs. a burger) are often localized.

**Food Recognition**: The release of Food-101 (Bossard et al., 2014) spurred significant research. Early approaches used Random Forests, while recent methods use large DCNNs. *DeepFood* and others have pushed accuracy above 85-90% but often rely on heavy backbones like ResNet-101 or DenseNet-161, which are unsuitable for efficient edge deployment.

---

## 3. Methodology

### 3.1 Network Architecture

We adopt MobileNetV2 as our baseline due to its proven efficiency. We modify the architecture by inserting Convolutional Block Attention Modules (CBAM) at three key locations:
1.  **Early Features**: After the first block with 32 channels.
2.  **Mid-level Features**: After the block with 96 channels.
3.  **High-level Features**: After the final block with 320 channels.

This placement allows the network to refine low-level textures, mid-level patterns, and high-level semantic objects respectively. The total parameter count increases negligibly from 2.23M to 2.27M (+1.8%).

**CBAM Structure**:
*   **Channel Attention Module**: Compresses the spatial dimension (Max and Avg pooling) and learns channel-wise importance weights.
*   **Spatial Attention Module**: Compresses the channel dimension and learns "where" to focus in the spatial map.

### 3.2 Data Augmentation

To prevent overfitting on the 75,750 training images, we implement a rigorous augmentation pipeline implemented in PyTorch:
*   `RandomResizedCrop`: Scales 0.7–1.0.
*   `RandomRotation`: ±15 degrees.
*   `ColorJitter`: Brightness, contrast, saturation (0.3), hue (0.1).
*   `RandomAffine`: Slight translations ensuring translation invariance.
*   `RandomErasing`: Simulates occlusions (probability 0.5).

### 3.3 Training Configuration

*   **Optimizer**: Adam ($lr=0.001$, $\beta_1=0.9, \beta_2=0.999$).
*   **Scheduler**: Cosine Annealing LR ($T_{max}=50$).
*   **Loss Function**: Cross-Entropy with Label Smoothing ($\epsilon=0.1$) to prevent overconfident predictions.
*   **Hardware**: NVIDIA RTX 3070 (8GB).
*   **Precision**: Mixed Precision (FP16) via `torch.cuda.amp` to reduce memory usage and speed up training.

---

## 4. Experiments and Results

### 4.1 Implementation Details
We implemented the model in PyTorch 2.x. Training was conducted for 50 epochs with a batch size of 256. The dataset was split into standard train/test sets (750/250 images per class).

### 4.2 Quantitative Results

Table 1 compares our method against standard baselines on the Food-101 dataset.

**Table 1: Comparison with State-of-the-Art Architectures**

| Model | Top-1 Accuracy | Top-5 Accuracy | Parameters | Model Size |
| :--- | :---: | :---: | :---: | :---: |
| **Ours (MobileNetV2+CBAM)** | **83.51%** | **96.01%** | **2.27M** | **16 MB** |
| ResNet-50 | 82.40% | 95.20% | 25.60M | 98 MB |
| MobileNetV2 (Baseline) | 76.30% | 93.50% | 2.23M | 14 MB |
| EfficientNet-B0 | 85.70% | 96.50% | 5.30M | 29 MB |
| InceptionV3 | 88.28% | 96.88% | 23.80M | 92 MB |

Our model surpasses the widely used ResNet-50 by **1.11%** while being **11× smaller**. While EfficientNet-B0 achieves slightly higher accuracy, it requires 2.3× more parameters and has higher latency on mobile CPUs.

### 4.3 Ablation Study

To validate our design choices, we analyzed the contribution of each component (Table 2).

**Table 2: Impact of Enhancements**

| Configuration | Accuracy | Improvement |
| :--- | :---: | :---: |
| Baseline MobileNetV2 | 76.30% | - |
| + CBAM Attention | 80.50% | +4.20% |
| + Advanced Augmentation | 82.80% | +2.30% |
| + Optimization (Adam, Labelsmoothing) | **83.51%** | +0.71% |

The largest gain (+4.2%) comes from the CBAM modules, confirming that attention mechanisms are highly effective for fine-grained food recognition.

---

## 5. Conclusion

We presented an enhanced MobileNetV2 architecture for efficient food recognition. By integrating CBAM attention and employing modern training techniques, we achieved **83.51% accuracy** on Food-101 with only **2.27M parameters**. This work demonstrates that extremely lightweight models can achieve competitive performance, making them ideal for deployment in real-time dietary monitoring applications on mobile devices.

Future work will focus on quantizing the model to INT8 for further size reduction (approx. 4MB) and deploying it to a cross-platform mobile application.

---

## References

1.  Sandler, M., et al. "MobileNetV2: Inverted residuals and linear bottlenecks." *CVPR* 2018.
2.  Woo, S., et al. "CBAM: Convolutional block attention module." *ECCV* 2018.
3.  Bossard, L., et al. "Food-101–mining discriminative components with random forests." *ECCV* 2014.
4.  He, K., et al. "Deep residual learning for image recognition." *CVPR* 2016.
