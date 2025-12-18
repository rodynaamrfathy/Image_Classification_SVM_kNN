import cv2
import numpy as np
import pickle
import joblib
import os
from src.features import extract_features

# ==============================
# Paths
# ==============================
MODEL_PATH = "/Users/rodynaamr/Image_Classification_SVM_kNN/src/svm_waste_classifier.pkl"
SCALER_PATH = "/Users/rodynaamr/Image_Classification_SVM_kNN/src/scaler.pkl"
LABEL_ENCODER_PATH = "/Users/rodynaamr/Image_Classification_SVM_kNN/src/label_encoder.pkl"

IMAGE_PATH = "/Users/rodynaamr/Image_Classification_SVM_kNN/data/cardboard/6ea71282-473c-4bd4-995e-520d20b43ea2.jpg"
CONFIDENCE_THRESHOLD = 0.60


def load_model():
    """Load SVM, scaler, and label encoder"""
    svm = joblib.load(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return svm, scaler, label_encoder


def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # Load model components
    svm, scaler, label_encoder = load_model()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    # Match training preprocessing
    image = cv2.resize(image, (94, 94))

    # Extract features
    features = extract_features(image)
    features = features.reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probabilities
    probs = svm.predict_proba(features_scaled)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    # Apply rejection
    if confidence < CONFIDENCE_THRESHOLD:
        label = "unknown"
    else:
        label = label_encoder.inverse_transform([pred_idx])[0]

    # ==============================
    # Output
    # ==============================
    print("=" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nClass probabilities:")
    for cls, p in zip(label_encoder.classes_, probs):
        print(f"  {cls:<10}: {p * 100:.2f}%")
    print("=" * 50)

    return label, confidence, probs


if __name__ == "__main__":
    predict_image(IMAGE_PATH)
