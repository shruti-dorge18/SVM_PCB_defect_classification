from pathlib import Path
import sys

import cv2
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "hog_svm_grouped_model.joblib"
CONFIG_PATH = MODEL_DIR / "hog_grouped_config.joblib"

if len(sys.argv) != 2:
    print('Usage: python predict_svm.py "/path/to/pcb_image.png"')
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"Error: Image not found: {image_path}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
config = joblib.load(CONFIG_PATH)

image_size = tuple(config["image_size"])
classes = config["classes"]

hog = cv2.HOGDescriptor(
    _winSize=image_size,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Could not read the image.")
    sys.exit(1)

image = cv2.resize(image, image_size)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
features = hog.compute(gray).flatten().reshape(1, -1)

prediction = model.predict(features)[0]
print(f"Prediction: {classes[prediction].replace('_', ' ').title()}")

if hasattr(model, "decision_function"):
    score = model.decision_function(features)[0]
    print(f"SVM decision score: {score:.4f}")
