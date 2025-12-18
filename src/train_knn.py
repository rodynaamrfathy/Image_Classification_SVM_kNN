import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =============================
# 1. Load features
# =============================

SAVE_DIR = "."
X_PATH = os.path.join(SAVE_DIR, "features.npy")
Y_PATH = os.path.join(SAVE_DIR, "labels.npy")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Loaded feature matrix shape:", X.shape)
print("Loaded labels shape:", y.shape)

# =============================
# 2. Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=62, stratify=y
)

# =============================
# 3. Train KNN
# =============================
knn = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',
    metric='euclidean'
)
knn.fit(X_train, y_train)

# =============================
# 4. Predict with rejection
# =============================
# Get distances and indices of neighbors
distances, indices = knn.kneighbors(X_test)

# Compute a rejection threshold (e.g., mean distance + 2*std)
threshold = np.mean(distances) + 2 * np.std(distances)
print("Rejection threshold:", threshold)

# Custom prediction with rejection
y_pred = []
for sample, dist in zip(X_test, distances):
    if np.min(dist) > threshold:  # too far from known points
        y_pred.append(6)         # -1 = unknown class
    else:
        y_pred.append(knn.predict([sample])[0])
y_pred = np.array(y_pred)

# Print number of unknowns
num_unknowns = np.sum(y_pred == 6)
print(f"\nNumber of unknown predictions: {num_unknowns}")

# Only evaluate known predictions
mask = y_pred != 6
acc = accuracy_score(y_test[mask], y_pred[mask])
print("\nAccuracy on known classes:", acc)

print("\nClassification Report (known classes only):")
print(classification_report(y_test[mask], y_pred[mask]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test[mask], y_pred[mask]))

# =============================
# 5. Save model
# =============================
joblib.dump(knn, "knn_waste_classifier.pkl")
print("\nModel saved as knn_waste_classifier.pkl")
