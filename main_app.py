import cv2
import numpy as np
import joblib
import time
from PIL import Image
from img2vec_pytorch import Img2Vec

# ==============================
# Paths (NEW)
# ==============================
MODEL_PATH = "src/models/svm_model.pkl"
LABEL_ENCODER_PATH = "src/models/label_encoder.pkl"

# ==============================
# Configuration
# ==============================
CONFIDENCE_THRESHOLD = 0.60
FRAME_SKIP = 10
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

WASTE_COLORS = {
    'cardboard': (139, 69, 19),
    'glass': (0, 255, 255),
    'metal': (128, 128, 128),
    'paper': (255, 255, 255),
    'plastic': (0, 165, 255),
    'trash': (0, 0, 255),
    'unknown': (128, 0, 128)
}


class WasteClassifier:

    def __init__(self,
                 model_path=MODEL_PATH,
                 label_encoder_path=LABEL_ENCODER_PATH,
                 confidence_threshold=CONFIDENCE_THRESHOLD):

        print("Loading SVM + Img2Vec model...")

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

        self.img2vec = Img2Vec(cuda=False)

        self.classes = self.label_encoder.classes_
        self.confidence_threshold = confidence_threshold

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        print("✓ Model loaded")
        print("✓ Classes:", self.classes)

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, frame):
        try:
            # OpenCV → PIL
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # CNN embedding
            vec = self.img2vec.get_vec(img)
            vec = np.array(vec).reshape(1, -1)

            # SVM prediction
            probs = self.model.predict_proba(vec)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]

            if confidence < self.confidence_threshold:
                return "unknown", confidence, probs

            label = self.label_encoder.inverse_transform([pred_idx])[0]
            return label, confidence, probs

        except Exception as e:
            print("Prediction error:", e)
            return None, 0.0, None

    # -----------------------------
    # FPS
    # -----------------------------
    def update_fps(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

    # -----------------------------
    # UI
    # -----------------------------
    def draw_ui(self, frame, waste_type, confidence, probabilities):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (10, 10), (w - 10, 180), (0, 0, 0), -1)

        cv2.putText(frame, "Real-Time Waste Classification (SVM)",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        if waste_type:
            color = WASTE_COLORS.get(waste_type, (255, 255, 255))
            cv2.putText(frame, f"Type: {waste_type.upper()}",
                        (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    # -----------------------------
    # Camera loop
    # -----------------------------
    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

        frame_counter = 0
        current_prediction = None
        current_confidence = 0.0
        current_probs = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % FRAME_SKIP == 0:
                current_prediction, current_confidence, current_probs = self.predict(frame)

            frame = self.draw_ui(frame, current_prediction, current_confidence, current_probs)
            cv2.imshow("Waste Classifier", frame)

            self.update_fps()
            frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    classifier = WasteClassifier()
    classifier.run(camera_id=0)
