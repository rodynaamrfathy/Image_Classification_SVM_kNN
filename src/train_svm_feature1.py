# ======================================
# 0. Imports
# ======================================
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib

# ======================================
# 1. Load Features
# ======================================
X = np.load("features.npy")
y = np.load("labels.npy")

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)
print("Classes:", np.unique(y))

# ======================================
# 2. Train/Test Split
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================
# 3. Train Base SVM
# ======================================
base_svm = SVC(
    kernel='rbf',
    C=10,
    gamma='auto',
    class_weight='balanced'
)

# ======================================
# 4. Calibrate the SVM
# ======================================
calibrated_svm = CalibratedClassifierCV(
    base_svm,
    method='sigmoid',
    cv=5
)

calibrated_svm.fit(X_train, y_train)
print("\nCalibrated SVM Training Completed!")

# ======================================
# 5. Unknown Class Prediction
# ======================================
CONFIDENCE_THRESHOLD = 0.60

def svm_predict_with_rejection(model, x, threshold=CONFIDENCE_THRESHOLD):
    probs = model.predict_proba([x])[0]
    pred = np.argmax(probs)
    confidence = probs[pred]

    if confidence < threshold:
        return 6  # unknown class index

    return pred

# ======================================
# 6. Evaluate
# ======================================
y_pred = np.array([
    svm_predict_with_rejection(calibrated_svm, x)
    for x in X_test
])

print("\n===== CALIBRATED SVM RESULTS (with Unknown class) =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ======================================
# 7. Save Calibrated Model
# ======================================
joblib.dump(calibrated_svm, "svm_waste_classifier.pkl")
print("\n Calibrated SVM model saved as svm_waste_classifier.pkl")
