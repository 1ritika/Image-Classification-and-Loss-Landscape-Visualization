# 🖼️ Image Classification and Loss-Landscape Visualization

This project explores **image classification** on the **CIFAR-10 dataset** using different CNN architectures and visualizes their **loss landscapes** to study optimization and generalization behavior.

---

## 🚀 Features
- Implemented and trained **PlainNet**, **CNN**, and **ResNet** (depth 20, 56, 110) models from scratch in PyTorch.  
- Compared convergence, training dynamics, and test accuracy across model depths.  
- Achieved **93.61% test accuracy** with **ResNet-110**, outperforming CNN baselines.  
- Generated **3D loss surface and contour visualizations** using **filter-wise normalization**, illustrating flatter minima for ResNet models.  

---

## 🧩 Dataset
- **Dataset:** CIFAR-10 (60,000 color images across 10 classes).  
- **Split:** 50,000 training / 10,000 test images.  
- **Augmentation:** Random cropping and flipping for better generalization.

---

## 📈 Results Summary
| Model | Depth | Test Accuracy | Observations |
|--------|--------|----------------|---------------|
| PlainNet | 20 | 85.12% | Struggles with vanishing gradients |
| CNN | - | 83.15% | Simple baseline for comparison |
| ResNet | **110** | **93.61%** | Flatter minima, better generalization |

> ResNet architectures show smoother loss landscapes and improved stability during optimization.

---

## 🛠️ Requirements
```bash
pip install torch torchvision matplotlib numpy
