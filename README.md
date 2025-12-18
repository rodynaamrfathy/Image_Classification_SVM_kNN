# ♻️ Image Classification: SVM & k-NN Waste Sorting System

An end-to-end machine learning project for automated waste material classification using Support Vector Machines (SVM) and k-Nearest Neighbors (k-NN). This system helps automate recycling processes by identifying different types of waste materials from images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green.svg)](https://opencv.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Material Classes](#-material-classes)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Material Stream Identification (MSI) System** that addresses the challenge of automated waste sorting—a critical bottleneck in achieving circular economy goals. The system uses computer vision and classical machine learning algorithms to classify waste materials into different categories.

**Key Objectives:**
- Design and implement a feature-based vision system for waste classification
- Train and compare two ML classifiers: SVM and k-NN
- Achieve minimum validation accuracy of 85% across primary classes
- Deploy a real-time classification application using live camera feed
- Implement data augmentation to improve model robustness

---

## ✨ Features

- **Real-time Classification**: Live camera feed processing with immediate predictions
- **Dual Model Comparison**: Side-by-side comparison of SVM and k-NN classifiers
- **Data Augmentation**: Automated dataset expansion (minimum 30% increase)
- **Feature Extraction**: Multiple feature engineering approaches including:
  - Image embeddings using Img2Vec (ResNet-18 based)
  - Custom feature extraction pipelines
- **Confidence-based Rejection**: Unknown class detection for out-of-distribution samples
- **Interactive Notebooks**: Step-by-step Jupyter notebooks for exploration and training
- **Production-ready Application**: Real-time deployment with OpenCV integration

---

## 📦 Material Classes

The system classifies waste into **7 distinct categories**:

| ID | Class | Description | Visual Identifier |
|----|-------|-------------|-------------------|
| 0 | **Glass** | Bottles, jars, glass containers | 🟡 Yellow |
| 1 | **Paper** | Newspapers, office paper, magazines | ⚪ White |
| 2 | **Cardboard** | Boxes, corrugated materials | 🟤 Brown |
| 3 | **Plastic** | Bottles, containers, packaging | 🟠 Orange |
| 4 | **Metal** | Aluminum cans, steel scrap | ⚫ Gray |
| 5 | **Trash** | Non-recyclable/contaminated waste | 🔴 Red |
| 6 | **Unknown** | Out-of-distribution items | 🟣 Purple |

---

## 📂 Project Structure

```
Image_Classification_SVM_kNN/
│
├── 📁 data/                      # Dataset directories
│   ├── cardboard/                # Original training images
│   ├── cardboard_aug/            # Augmented images
│   ├── glass/
│   ├── glass_aug/
│   ├── metal/
│   ├── metal_aug/
│   ├── paper/
│   ├── paper_aug/
│   ├── plastic/
│   ├── plastic_aug/
│   ├── trash/
│   └── trash_aug/
│
├── 📁 notebooks/                 # Jupyter notebooks for experimentation
│   ├── 01_explore_data.ipynb        # Data exploration & visualization
│   ├── 02_data_augmentation.ipynb   # Data augmentation pipeline
│   ├── 03_feature_extraction.ipynb  # Feature engineering
│   ├── 04_train_svm_knn.ipynb       # Model training
│   └── 05_evaluation.ipynb          # Model evaluation & metrics
│
├── 📁 src/                       # Source code
│   ├── features.py                  # Feature extraction pipeline
│   ├── features_Imp2vec.py          # Img2Vec-based features
│   ├── train_svm_feature1.py        # SVM training (approach 1)
│   ├── train_svm_feature2.py        # SVM training (approach 2)
│   ├── train_knn.py                 # k-NN training
│   ├── inference.py                 # Inference & rejection mechanism
│   ├── utils.py                     # Utility functions
│   └── 📁 models/                   # Saved models & features
│       ├── svm_model.pkl
│       ├── label_encoder.pkl
│       ├── features.npy
│       └── labels.npy
│
├── 📁 checkpoints/               # Model checkpoints
├── 📁 logs/                      # Training logs
│
├── main_app.py                   # 🚀 Real-time classification app
├── requirements.txt              # Python dependencies
└── README.md                     # This file

```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time application)
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/rodynaamrfathy/Image_Classification_SVM_kNN.git
   cd Image_Classification_SVM_kNN
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset** (if not included)
   - Place images in respective folders under `data/`
   - Ensure each material class has its own directory

---

## 🚀 Usage

### 1. Data Exploration & Augmentation

Start with the Jupyter notebooks to understand the dataset:

```bash
jupyter notebook notebooks/01_explore_data.ipynb
```

Run data augmentation to expand the training set:

```bash
jupyter notebook notebooks/02_data_augmentation.ipynb
```

### 2. Feature Extraction

Extract features from images:

```bash
python src/features.py
# or use Img2Vec approach
python src/features_Imp2vec.py
```

### 3. Train Models

**Train SVM:**
```bash
python src/train_svm_feature1.py
# or
python src/train_svm_feature2.py
```

**Train k-NN:**
```bash
python src/train_knn.py
```

### 4. Run Real-time Application

Launch the live camera classification system:

```bash
python main_app.py
```

**Application Controls:**
- Press `q` to quit
- The application displays:
  - Live camera feed
  - Predicted class with confidence score
  - Color-coded bounding box for each material type

---

## 🔬 Methodology

### 1. Data Augmentation

To improve model generalization, the dataset is augmented using:
- **Rotation**: Random rotations (±15-30 degrees)
- **Flipping**: Horizontal and vertical flips
- **Scaling**: Zoom in/out transformations
- **Color Jitter**: Brightness and contrast adjustments
- **Noise Injection**: Adding Gaussian noise

This increases the training set by **minimum 30%**, making models more robust to variations.

### 2. Feature Extraction

Two primary approaches:

**Approach 1: Deep Learning Embeddings**
- Uses pre-trained ResNet-18 via `Img2Vec`
- Extracts 512-dimensional feature vectors
- Transfer learning from ImageNet

**Approach 2: Custom Feature Engineering**
- Color histograms
- Texture descriptors (LBP, HOG)
- Shape features
- Statistical moments

### 3. Classification Algorithms

**Support Vector Machine (SVM)**
- Kernel: RBF (Radial Basis Function)
- Hyperparameter tuning via grid search
- Optimal C and gamma values

**k-Nearest Neighbors (k-NN)**
- Distance metric: Euclidean
- Weighting: Distance-based
- Optimal k value selection

### 4. Rejection Mechanism

Implements confidence threshold for the "Unknown" class:
- If prediction confidence < 60%, classify as "Unknown"
- Prevents misclassification of out-of-distribution samples
- Configurable threshold in `main_app.py`

---

## 📊 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (RBF) | 88.5% | 87.3% | 88.1% | 87.7% |
| k-NN (k=5) | 85.2% | 84.6% | 85.0% | 84.8% |

### Key Findings

- ✅ Both models exceed the 85% accuracy target
- ✅ SVM performs slightly better with complex feature representations
- ✅ k-NN offers faster inference time for real-time applications
- ✅ Data augmentation improved accuracy by ~7-10%
- ✅ Img2Vec features outperform hand-crafted features

### Real-time Performance

- **FPS**: 10-15 frames per second (with frame skipping)
- **Latency**: ~100-150ms per classification
- **Confidence threshold**: 60% (configurable)

---

## 🛠️ Configuration

Edit `main_app.py` to customize:

```python
CONFIDENCE_THRESHOLD = 0.60    # Minimum confidence for classification
FRAME_SKIP = 10                # Process every Nth frame
DISPLAY_WIDTH = 1280           # Display resolution
DISPLAY_HEIGHT = 720
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Rodyna Amr Fathy** - [GitHub](https://github.com/rodynaamrfathy)

---

## 🙏 Acknowledgments

- Dataset sourced from recycling material databases
- Pre-trained models from PyTorch Model Zoo
- Inspired by circular economy and sustainability initiatives

---

## 📧 Contact

For questions or collaborations:
- GitHub Issues: [Create an issue](https://github.com/rodynaamrfathy/Image_Classification_SVM_kNN/issues)
- Email: Contact via GitHub profile

---

**⭐ If you find this project helpful, please consider giving it a star!**

