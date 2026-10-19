from pathlib import Path

import cv2
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_grouped"
MODEL_DIR = BASE_DIR / "models"

CLASSES = ["defective", "non_defective"]

config = joblib.load(MODEL_DIR / "hog_grouped_config.joblib")
model = joblib.load(MODEL_DIR / "hog_svm_grouped_model.joblib")

IMAGE_SIZE = tuple(config["image_size"])

hog = cv2.HOGDescriptor(
    _winSize=IMAGE_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def extract_hog(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return hog.compute(gray).flatten()


def evaluate_split(split_name):
    split_dir = DATASET_DIR / split_name
    features = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        folder = split_dir / class_name

        image_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        print(f"{split_name}/{class_name}: {len(image_paths)} images")

        for image_path in image_paths:
            features.append(extract_hog(image_path))
            labels.append(label)

    if not labels:
        print(f"No images found for {split_name}.")
        return

    X = np.asarray(features)
    y_true = np.asarray(labels)
    y_pred = model.predict(X)

    print(f"\n{'=' * 50}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'=' * 50}")

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASSES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Rows = Actual, Columns = Predicted")
    print("Column order:", CLASSES)
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    evaluate_split("validation")
    evaluate_split("test")
