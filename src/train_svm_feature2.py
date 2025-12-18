import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Paths
SAVE_DIR = "models"
X_PATH = os.path.join(SAVE_DIR, "features.npy")
Y_PATH = os.path.join(SAVE_DIR, "labels.npy")

# Load data
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Loaded features:", X.shape)
print("Loaded labels:", y.shape)

# Train / validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM pipeline
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    ))
])

# Train
print("Training SVM...")
svm_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = svm_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm_pipeline, os.path.join(SAVE_DIR, "svm_model.pkl"))

print(" ---- SVM model saved ----")
