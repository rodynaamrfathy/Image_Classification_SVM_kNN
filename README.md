## ♻️ ML Project: Material Stream Identification (MSI) System

[cite_start]This repository contains the full implementation of the **Material Stream Identification (MSI) System**, a machine learning project for the automated sorting of post-consumer waste[cite: 17]. [cite_start]This system addresses a critical bottleneck in achieving circular economy goals by developing a feature-based vision system for automated waste classification[cite: 16, 20].

[cite_start]The project focuses on mastering the entire ML pipeline: Data Preprocessing, Feature Extraction, Classifier Training, and Performance Evaluation[cite: 18].

-----

## 🚀 Project Objectives

[cite_start]The primary goal is the end-to-end implementation and comparison of two foundational machine learning classifiers for waste material identification[cite: 20, 23].

  * [cite_start]**Feature-Based Vision System:** Design and implement a pipeline to convert raw images into a fixed-size, numerical **feature vector**[cite: 21, 22].
  * [cite_start]**Classifier Training:** Implement and train two distinct classifiers on the extracted feature vectors for the seven material categories[cite: 23].
      * [cite_start]**Variant A:** Support Vector Machine (**SVM**)[cite: 24].
      * [cite_start]**Variant B:** k-Nearest Neighbors (**k-NN**)[cite: 25].
  * [cite_start]**Performance:** Achieve a minimum validation accuracy of **0.85** across the six primary classes[cite: 27].
  * [cite_start]**Deployment:** Integrate the best-performing model into a functional application that processes **live camera frames in real-time**[cite: 28, 56].
  * [cite_start]**Analysis:** Conduct an architecture comparison, analyzing the trade-offs between the two classifiers and the different chosen feature extraction methods[cite: 26].

-----

## 📦 Material Classes

[cite_start]The system classifies input images into one of seven classes[cite: 31]:

| ID | Common Name | Description |
| :---: | :---: | :--- |
| **0** | **Glass** | [cite_start]Items made of amorphous solid materials, primarily silicates (e.g., bottles, jars)[cite: 32]. |
| **1** | **Paper** | [cite_start]Thin materials made from pressed cellulose pulp (e.g., newspapers, office paper)[cite: 32]. |
| **2** | **Cardboard** | [cite_start]Heavy-duty structured material composed of multiple layers of cellulose fiber[cite: 32]. |
| **3** | **Plastic** | [cite_start]Items made from high-molecular-weight organic compounds (e.g., water bottles, film)[cite: 32]. |
| **4** | **Metal** | [cite_start]Items made of elemental or compound metallic substances (e.g., aluminum cans, steel scrap)[cite: 32]. |
| **5** | **Trash** | [cite_start]Miscellaneous non-recyclable or contaminated waste (e.g., organic matter, food packaging)[cite: 32]. |
| **6** | **Unknown** | [cite_start]Mandatory class for out-of-distribution items or blurred inputs; requires a rejection mechanism[cite: 32, 53, 54]. |

-----

## 💻 Technical Requirements and Implementation

### 1\. Data Augmentation

[cite_start]To improve model generalization and robustness, data augmentation techniques are applied to the dataset[cite: 39, 40]. [cite_start]The training sample size is artificially increased by a **minimum of 30%**[cite: 39]. [cite_start]Techniques used (e.g., rotation, flipping, scaling, and color jitter) are documented in the technical report[cite: 41].

### 2\. Feature Extraction

[cite_start]A core part of the project is converting the raw image data into a **1D numerical feature vector**[cite: 43]. [cite_start]The methodology and choice of feature descriptors (e.g., color histograms, texture features like LBP, or structural features like HOG) are implemented in `src/features.py` and justified in the report[cite: 44].

### 3\. Model Architecture and Training

[cite_start]The classifiers are implemented to accept the extracted feature vector as input[cite: 50, 52].

  * **SVM:** Implemented in `src/train_svm.py`. [cite_start]Requires selection and justification of the optimal architecture elements, such as the kernel (e.g., RBF, linear)[cite: 51].
  * **k-NN:** Implemented in `src/train_knn.py`. [cite_start]Requires selecting and justifying the weighting scheme (e.g., uniform, distance-based)[cite: 52].

### 4\. Handling the 'Unknown' Class

[cite_start]A **rejection mechanism** is implemented for both models (in `src/inference.py`) to handle the 'Unknown' class (ID 6), ensuring the system only classifies items it is confident about[cite: 53, 54].

-----

## 📂 Repository Structure

```
.
├── checkpoints/             # Used to store temporary model saves during training
├── data/                    # Placeholder for the raw dataset
├── logs/                    # Training and experiment logs
├── models/                  # Final saved classifier weights for submission
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_data_augmentation.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_train_svm_knn.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── augment.py           # Data augmentation functions
│   ├── features.py          # Image-to-vector feature conversion
│   ├── inference.py         # Real-time classification logic and rejection mechanism
│   ├── train_knn.py         # k-NN model training script
│   ├── train_svm.py         # SVM model training script
│   └── utils.py             # Helper utilities
├── main_app.py              # Functional application for live camera deployment
└── requirements.txt         # Required Python dependencies
```

-----

