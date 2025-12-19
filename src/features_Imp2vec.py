import os
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = "/Users/rodynaamr/Image_Classification_SVM_kNN/data"
SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)

img2vec = Img2Vec()

X, y = [], []

for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # Map augmented folders to the original class name
    label = class_name.replace("_aug", "")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            img = Image.open(img_path).convert('RGB')  # ensure 3 channels
            vec = img2vec.get_vec(img)
            X.append(vec)
            y.append(label)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

X = np.array(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

np.save(os.path.join(SAVE_DIR, "features.npy"), X)
np.save(os.path.join(SAVE_DIR, "labels.npy"), y_encoded)
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))

print("Features shape:", X.shape)
print("Classes:", le.classes_)
