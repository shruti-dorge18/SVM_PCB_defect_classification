from pathlib import Path
import cv2
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_grouped" / "train"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (128, 128)
CLASSES = ["defective", "non_defective"]

# HOG feature extractor
hog = cv2.HOGDescriptor(
    _winSize=IMAGE_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)


def extract_hog(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog.compute(gray)
    return features.flatten()


def load_dataset():
    features = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        folder = DATASET_DIR / class_name
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        print(f"{class_name}: {len(image_paths)} images")

        for index, image_path in enumerate(image_paths):
            features.append(extract_hog(image_path))
            labels.append(label)

            if (index + 1) % 200 == 0:
                print(f"  Processed {index + 1}/{len(image_paths)}")

    return np.asarray(features), np.asarray(labels)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting HOG features from training images...")
    X, y = load_dataset()

    print("Feature matrix:", X.shape)
    print("Training SVM from scratch...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", class_weight="balanced"))
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    print("\nTraining accuracy:", f"{accuracy_score(y, predictions):.2%}")
    print("\nTraining report:")
    print(classification_report(y, predictions, target_names=CLASSES))

    model_path = MODEL_DIR / "hog_svm_grouped_model.joblib"
    joblib.dump(model, model_path)

    config_path = MODEL_DIR / "hog_grouped_config.joblib"
    joblib.dump({
        "image_size": IMAGE_SIZE,
        "classes": CLASSES,
        "feature_type": "HOG",
        "classifier": "Linear SVM"
    }, config_path)

    print(f"\nSVM model saved to: {model_path}")
    print(f"HOG configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
