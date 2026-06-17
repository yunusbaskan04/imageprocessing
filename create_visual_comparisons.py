import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

EVAL_IMAGES_DIR = os.path.join("dataset_eval", "images")
EVAL_LABELS_DIR = os.path.join("dataset_eval", "labels")
OUTPUT_BASE_DIR = "outputs"
FIGURES_DIR = os.path.join("results", "figures")

DEGRADATIONS = ["low_light", "fog", "noise"]

VISUAL_COLUMNS = [
    ("original", "Original"),
    ("degraded", "Degraded"),
    ("global_he", "Global HE"),
    ("clahe_only", "CLAHE-only"),
    ("proposed", "CLAHE+Bilateral"),
    ("morphology_ablation", "Proposed-Final"),
]

YOLO_COLUMNS = [
    ("original", "Original"),
    ("degraded", "Degraded"),
    ("proposed", "CLAHE+Bilateral"),
    ("morphology_ablation", "Proposed-Final"),
]

DEGRADATION_LABELS = {
    "low_light": "Low-light",
    "fog": "Fog",
    "noise": "Noise"
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# COCO aeroplane class id. This matches the zero-shot evaluation protocol.
TARGET_CLASS_ID = 4


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    if not os.path.exists(folder):
        return []
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]


def find_labelled_sample():
    """
    Selects the first image that has at least one annotation.
    If the output figure is visually weak, manually replace SAMPLE_IMAGE below.
    """
    for img_name in list_images(EVAL_IMAGES_DIR):
        stem, _ = os.path.splitext(img_name)
        label_path = os.path.join(EVAL_LABELS_DIR, stem + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if lines:
            return img_name

    images = list_images(EVAL_IMAGES_DIR)
    return images[0] if images else None


def read_rgb(path):
    if not os.path.exists(path):
        return None

    img = cv2.imread(path)
    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_image_path(method, degradation, sample_image):
    if method == "original":
        return os.path.join(EVAL_IMAGES_DIR, sample_image)

    return os.path.join(OUTPUT_BASE_DIR, method, degradation, sample_image)


def create_visual_comparison(sample_image):
    rows = len(DEGRADATIONS)
    cols = len(VISUAL_COLUMNS)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))

    for r, degradation in enumerate(DEGRADATIONS):
        for c, (method, title) in enumerate(VISUAL_COLUMNS):
            ax = axes[r][c]
            img_path = get_image_path(method, degradation, sample_image)
            img = read_rgb(img_path)

            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")

            if r == 0:
                ax.set_title(title, fontsize=10)

            if c == 0:
                ax.set_ylabel(DEGRADATION_LABELS[degradation], fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig6_visual_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def predict_rgb(model, img_path):
    results = model.predict(
        img_path,
        imgsz=640,
        conf=0.25,
        classes=[TARGET_CLASS_ID],
        verbose=False
    )

    plotted = results[0].plot()

    # Ultralytics plot output is usually BGR-like for OpenCV usage.
    # Convert for matplotlib display.
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return plotted_rgb


def create_yolo_visual_comparison(sample_image):
    print("Loading YOLO model for visual predictions...")
    model = YOLO(MODEL_PATH)

    rows = len(DEGRADATIONS)
    cols = len(YOLO_COLUMNS)

    fig, axes = plt.subplots(rows, cols, figsize=(13, 8))

    for r, degradation in enumerate(DEGRADATIONS):
        for c, (method, title) in enumerate(YOLO_COLUMNS):
            ax = axes[r][c]
            img_path = get_image_path(method, degradation, sample_image)

            if os.path.exists(img_path):
                try:
                    pred_img = predict_rgb(model, img_path)
                    ax.imshow(pred_img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Prediction error\n{e}", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")

            if r == 0:
                ax.set_title(title, fontsize=10)

            if c == 0:
                ax.set_ylabel(DEGRADATION_LABELS[degradation], fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig7_yolo_visual_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def main():
    ensure_dir(FIGURES_DIR)

    # Otomatik seçilen örnek görüntü.
    # Figür kötü çıkarsa buraya manuel dosya adı yazabiliriz:
    # SAMPLE_IMAGE = "valid_xxxxx.jpg"
    SAMPLE_IMAGE = "test_1-84-_jpeg.rf.26e95dfc944ffa124d8ca9fa876997c7.jpg"

    sample_image = SAMPLE_IMAGE or find_labelled_sample()

    if sample_image is None:
        print("No sample image found.")
        return

    print(f"Selected sample image: {sample_image}")

    create_visual_comparison(sample_image)
    create_yolo_visual_comparison(sample_image)

    print("\nVisual comparison figures completed.")


if __name__ == "__main__":
    main()