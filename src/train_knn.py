import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =============================
# 1. Load extracted features
# =============================
X = np.load("features.npy")
y = np.load("labels.npy")

print("Loaded feature matrix shape:", X.shape)
print("Loaded labels shape:", y.shape)

# =============================
# 2. Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 3. Create and train KNN model
# =============================
knn = KNeighborsClassifier(
    n_neighbors=5,      # you can tune this
    weights='distance', # helps performance
    metric='euclidean'  # default and good for hist+HOG
)

knn.fit(X_train, y_train)

# =============================
# 4. Evaluate
# =============================
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =============================
# 5. Save model
# =============================
joblib.dump(knn, "knn_waste_classifier.pkl")
print("\nModel saved as knn_waste_classifier.pkl")
