import sys
import os
import joblib
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) or "."
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "src", "models", "label_encoder.pkl")

# -----------------------------
# Load model & encoder
# -----------------------------
svm_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

img2vec = Img2Vec()


def predict_image(image_path, threshold=0.6):
    """Return predicted label (or "unknown") and max confidence."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    vec = img2vec.get_vec(img)
    vec = np.array(vec).reshape(1, -1)

    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(vec)[0]
        max_prob = float(probs.max())
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        if max_prob < threshold:
            return "unknown", max_prob
        return pred_label, max_prob

    pred_idx = svm_model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_label, None


def evaluate_dataset(dataset_dir, threshold=0.6):
    """Compute accuracy on a folder structured dataset_dir/class_name/*.jpg."""
    image_paths, true_labels = [], []

    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(cls_path, img_file))
                true_labels.append(cls)

    if not image_paths:
        raise ValueError(f"No images found under {dataset_dir}")

    known_classes = set(label_encoder.classes_)
    y_true, y_pred = [], []

    for path, true_label in zip(image_paths, true_labels):
        mapped_true = true_label if true_label in known_classes else "unknown"
        pred_label, conf = predict_image(path, threshold=threshold)
        y_true.append(mapped_true)
        y_pred.append(pred_label)

    from sklearn.metrics import classification_report, accuracy_score

    acc = accuracy_score(y_true, y_pred)
    print(f"Evaluated {len(image_paths)} images from {dataset_dir}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))
    return acc

# -----------------------------
# CLI usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate or predict with SVM model")
    parser.add_argument("path", help="Image path or dataset directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Unknown threshold when predict_proba is available")
    parser.add_argument("--mode", choices=["predict", "eval"], default="eval", help="predict single image or eval directory")
    args = parser.parse_args()

    if args.mode == "predict":
        label, conf = predict_image(args.path, threshold=args.threshold)
        if conf is None:
            print(f"Prediction: {label}")
        else:
            print(f"Prediction: {label} | Confidence: {conf:.4f}")
    else:
        evaluate_dataset(args.path, threshold=args.threshold)

