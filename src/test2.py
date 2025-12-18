import sys
import os
import joblib
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/svm_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# -----------------------------
# Load model & encoder
# -----------------------------
svm_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

img2vec = Img2Vec()

# -----------------------------
# Predict single image
# -----------------------------
def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")

    # Extract features
    vec = img2vec.get_vec(img)
    vec = np.array(vec).reshape(1, -1)

    # Predict
    pred_idx = svm_model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Optional confidence
    if hasattr(svm_model, "predict_proba"):
        confidence = svm_model.predict_proba(vec).max()
        print(f"Prediction: {pred_label} | Confidence: {confidence:.4f}")
    else:
        print(f"Prediction: {pred_label}")

    return pred_label


# -----------------------------
# CLI usage
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    predict_image(IMAGE_PATH)

