import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# ==============================
# Configuration
# ==============================
TARGET_SIZE = (128, 128)
COLOR_BINS = 64
HOG_PPC = (32, 32)
HOG_CPB = (2, 2)
HOG_ORIENT = 9
LBP_P = 16
LBP_R = 2


# ==============================
# Enhanced Feature Functions
# ==============================
def color_histogram(img, bins=COLOR_BINS):
    """Extract normalized color histogram features"""
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist = hist / (np.sum(hist) + 1e-7)
    return hist


def color_moments(img):
    """Extract color moments (mean, std, skewness) for each channel"""
    features = []
    for i in range(3):
        channel = img[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
        features.extend([mean, std, skewness])
    return np.array(features)


def edge_features(img):
    """Extract edge-based features to distinguish materials"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    mean_gradient = np.mean(gradient_magnitude)
    std_gradient = np.std(gradient_magnitude)

    return np.array([edge_density, mean_gradient, std_gradient])


def texture_glcm(img):
    """Extract GLCM texture features (contrast, homogeneity, energy)"""
    gray = rgb2gray(img)
    gray = img_as_ubyte(gray)

    # Compute GLCM at multiple angles and distances
    glcm = graycomatrix(gray, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)

    # Extract properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()

    return np.concatenate([contrast, homogeneity, energy, correlation])


def hog_features(img, pixels_per_cell=HOG_PPC, cells_per_block=HOG_CPB, orientations=HOG_ORIENT):
    """Extract HOG features from grayscale image"""
    gray = rgb2gray(img)
    features = hog(gray,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    return features


def lbp_features(img, P=LBP_P, R=LBP_R):
    """Extract Local Binary Pattern histogram features"""
    gray = rgb2gray(img)
    gray = img_as_ubyte(gray)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_features(img, target_size=TARGET_SIZE):
    """Extract ENHANCED combined feature vector"""
    img_resized = cv2.resize(img, target_size)

    # Original features
    color_feat = color_histogram(img_resized)
    hog_feat = hog_features(img_resized)
    lbp_feat = lbp_features(img_resized)

    # NEW discriminative features
    color_mom = color_moments(img_resized)
    edge_feat = edge_features(img_resized)
    glcm_feat = texture_glcm(img_resized)

    # Combine all features
    features = np.concatenate([
        color_feat,  # Color distribution
        color_mom,  # Color statistics
        hog_feat,  # Shape/structure
        lbp_feat,  # Local texture
        edge_feat,  # Edge characteristics
        glcm_feat  # Global texture patterns
    ])
    return features


# ==============================
# Main pipeline with balancing check
# ==============================
if __name__ == "__main__":
    base_path = "/Users/rodynaamr/Image_Classification_SVM_kNN/data"
    waste_types = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    all_features = []
    all_labels = []
    class_counts = {waste: 0 for waste in waste_types}

    for waste in waste_types:
        for folder_suffix in ["", "_aug"]:
            folder_path = os.path.join(base_path, waste + folder_suffix)
            if not os.path.exists(folder_path):
                continue

            for filename in tqdm(os.listdir(folder_path), desc=f"Processing {waste}{folder_suffix}"):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                feats = extract_features(img)

                all_features.append(feats)
                all_labels.append(waste)
                class_counts[waste] += 1

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    # Enhanced diagnostics
    print(f"\n=== Feature Extraction Summary ===")
    print(f"Total samples: {X.shape[0]}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"\nClass distribution:")
    for waste, count in sorted(class_counts.items(), key=lambda x: x[1]):
        percentage = (count / X.shape[0]) * 100
        print(f"  {waste:12s}: {count:5d} ({percentage:5.1f}%)")

    # Check for class imbalance
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    imbalance_ratio = max_samples / min_samples
    if imbalance_ratio > 2.0:
        print(f"\n⚠️  WARNING: Class imbalance detected! Ratio: {imbalance_ratio:.1f}x")
        print(f"   Consider balancing your dataset or using class_weight='balanced'")

    print(f"\nData quality checks:")
    print(f"  Any NaN?: {np.isnan(X).any()}")
    print(f"  Any inf?: {np.isinf(X).any()}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save everything
    np.save("features.npy", X_scaled)
    np.save("labels.npy", y_encoded)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"\n✓ Saved features, labels, scaler, and label encoder")
    print(f"✓ Classes: {le.classes_}")
    print(f"✓ Feature vector dimension increased from ~XXX to {X.shape[1]}")