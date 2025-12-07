import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =============================
# 1. Load features
# =============================
X = np.load("features.npy")
y = np.load("labels.npy")

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# =============================
# 2. Split the dataset
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 3. Train SVM model
# =============================
model = SVC(
    kernel='rbf',     # 'linear' or 'rbf'—rbf often better for image features
    C=10,             # regularization parameter (tune this)
    gamma='scale'     # auto scaling
)

model.fit(X_train, y_train)

# =============================
# 4. Evaluation
# =============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =============================
# 5. Save SVM model
# =============================
joblib.dump(model, "svm_waste_classifier.pkl")
print("\nSaved model: svm_waste_classifier.pkl")
