# inference.py
import os
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
import joblib

# Paths
SAVE_DIR = "models"
MODEL_PATH = os.path.join(SAVE_DIR, "svm_model.pkl")
LE_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")

# Initialize Img2Vec
img2vec = Img2Vec()

# Load trained model and label encoder
svm_pipeline = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)

def predict_image(img_path: str, threshold: float = 0.6):
    try:
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None, None

    vec = img2vec.get_vec(img)
    vec = np.array(vec).reshape(1, -1)

    pred_proba = svm_pipeline.predict_proba(vec)[0]
    max_prob = max(pred_proba)

    if max_prob < threshold:
        return "unknown", {cls: prob for cls, prob in zip(le.classes_, pred_proba)}

    pred_label_encoded = svm_pipeline.predict(vec)[0]
    pred_label = le.inverse_transform([pred_label_encoded])[0]
    proba_dict = {cls: prob for cls, prob in zip(le.classes_, pred_proba)}

    return pred_label, proba_dict


if __name__ == "__main__":
    # Example usage
    test_image = "/Users/rodynaamr/Downloads/mouse.png"
    label, probs = predict_image(test_image, threshold=0.6)
    if label:
        print(f"Predicted label: {label}")
        print("Probabilities:", probs)
