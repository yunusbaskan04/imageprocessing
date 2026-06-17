import os
import csv
import shutil
import yaml
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

EVAL_IMAGES_DIR = os.path.join("dataset_eval", "images")
EVAL_LABELS_DIR = os.path.join("dataset_eval", "labels")

OUTPUT_BASE_DIR = "outputs"
RESULTS_DIR = "results"
VISUALS_DIR = os.path.join(RESULTS_DIR, "yolo_visuals")

TEMP_EVAL_DIR = "temp_eval"

DEGRADATIONS = ["low_light", "fog", "noise"]

METHODS = [
    "degraded",
    "global_he",
    "clahe_only",
    "proposed",
    "morphology_ablation"
]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# COCO-pretrained YOLOv8n does not include a drone class.
# The closest aerial COCO class is aeroplane, whose class id is 4.
TARGET_CLASS_ID = 4
TARGET_CLASS_NAME = "aeroplane"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    if not os.path.exists(folder):
        return []

    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]


def setup_eval_env(images_src, labels_src, target_class_id=4):
    """
    Creates a temporary YOLO evaluation directory.

    Original UAV labels use class id 0 (drone).
    Since the pretrained COCO YOLOv8n model has no drone class,
    the label class id is mapped to COCO aeroplane class id 4.
    The same mapping is applied to all methods for a fair zero-shot comparison.
    """
    if os.path.exists(TEMP_EVAL_DIR):
        shutil.rmtree(TEMP_EVAL_DIR)

    images_dir = os.path.join(TEMP_EVAL_DIR, "images")
    labels_dir = os.path.join(TEMP_EVAL_DIR, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_files = list_images(images_src)

    for filename in image_files:
        shutil.copy2(
            os.path.join(images_src, filename),
            os.path.join(images_dir, filename)
        )

        stem, _ = os.path.splitext(filename)
        src_label = os.path.join(labels_src, stem + ".txt")
        dst_label = os.path.join(labels_dir, stem + ".txt")

        if os.path.exists(src_label):
            with open(src_label, "r", encoding="utf-8") as f_in, open(dst_label, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    parts = line.strip().split()

                    if len(parts) >= 5:
                        parts[0] = str(target_class_id)
                        f_out.write(" ".join(parts) + "\n")
                    else:
                        # Preserve empty or malformed lines safely by ignoring them.
                        continue
        else:
            # If label does not exist, create an empty file.
            open(dst_label, "w", encoding="utf-8").close()

    names_dict = {i: f"class_{i}" for i in range(80)}
    names_dict[target_class_id] = TARGET_CLASS_NAME

    data_yaml = {
        "path": os.path.abspath(TEMP_EVAL_DIR),
        "train": "images",
        "val": "images",
        "nc": 80,
        "names": names_dict
    }

    yaml_path = os.path.join(TEMP_EVAL_DIR, "data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    return yaml_path, len(image_files)


def find_sample_images(images_src, labels_src, max_samples=2):
    """
    Selects sample images that have at least one non-empty label.
    These are used only for visual bounding-box examples.
    """
    samples = []

    for filename in list_images(images_src):
        stem, _ = os.path.splitext(filename)
        label_path = os.path.join(labels_src, stem + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if lines:
            samples.append(filename)

        if len(samples) >= max_samples:
            break

    # Fallback: use first images if no labelled sample is found.
    if not samples:
        samples = list_images(images_src)[:max_samples]

    return samples


def save_visual_predictions(model, images_src, labels_src, degradation, method):
    """
    Saves a few YOLO prediction images for qualitative comparison.
    """
    ensure_dir(VISUALS_DIR)

    sample_images = find_sample_images(images_src, labels_src, max_samples=2)

    for img_name in sample_images:
        img_path = os.path.join(images_src, img_name)

        results = model.predict(
            img_path,
            imgsz=640,
            conf=0.25,
            classes=[TARGET_CLASS_ID],
            verbose=False
        )

        plotted = results[0].plot()

        safe_name = f"{degradation}_{method}_{img_name}"
        out_path = os.path.join(VISUALS_DIR, safe_name)

        cv2.imwrite(out_path, plotted)


def evaluate_dataset(model, degradation, method, images_src, labels_src):
    print("\n" + "=" * 90)
    print(f"Evaluating: degradation={degradation}, method={method}")
    print("=" * 90)

    yaml_path, image_count = setup_eval_env(
        images_src,
        labels_src,
        target_class_id=TARGET_CLASS_ID
    )

    metrics = model.val(
        data=yaml_path,
        imgsz=640,
        classes=[TARGET_CLASS_ID],
        verbose=False,
        plots=False
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    save_visual_predictions(
        model,
        images_src,
        labels_src,
        degradation,
        method
    )

    row = {
        "degradation": degradation,
        "method": method,
        "image_count": image_count,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95
    }

    print(
        f"Images: {image_count} | "
        f"mAP@50: {map50:.4f} | "
        f"mAP@50-95: {map50_95:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )

    return row


def save_csv(rows, csv_path):
    fieldnames = [
        "degradation",
        "method",
        "image_count",
        "precision",
        "recall",
        "map50",
        "map50_95"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_summary_table(rows):
    print("\n" + "=" * 110)
    print("YOLOv8n ZERO-SHOT DETECTION RESULTS")
    print("=" * 110)
    print(
        f"| {'Degradation':<12} | {'Method':<20} | {'Images':<6} | "
        f"{'mAP@50':<10} | {'mAP@50-95':<10} | {'Precision':<10} | {'Recall':<10} |"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"| {row['degradation']:<12} | "
            f"{row['method']:<20} | "
            f"{row['image_count']:<6} | "
            f"{row['map50']:<10.4f} | "
            f"{row['map50_95']:<10.4f} | "
            f"{row['precision']:<10.4f} | "
            f"{row['recall']:<10.4f} |"
        )

    print("=" * 110)


def main():
    ensure_dir(RESULTS_DIR)
    ensure_dir(VISUALS_DIR)

    if not os.path.exists(EVAL_IMAGES_DIR):
        print(f"ERROR: Evaluation images folder not found: {EVAL_IMAGES_DIR}")
        print("First run prepare_eval_subset.py.")
        return

    if not os.path.exists(EVAL_LABELS_DIR):
        print(f"ERROR: Evaluation labels folder not found: {EVAL_LABELS_DIR}")
        print("First run prepare_eval_subset.py.")
        return

    print("=" * 90)
    print("Loading YOLOv8n model")
    print("=" * 90)

    model = YOLO(MODEL_PATH)

    rows = []

    # Clean/original zero-shot reference
    rows.append(
        evaluate_dataset(
            model=model,
            degradation="clean",
            method="original",
            images_src=EVAL_IMAGES_DIR,
            labels_src=EVAL_LABELS_DIR
        )
    )

    # Degraded + baseline/proposed methods
    for degradation in DEGRADATIONS:
        for method in METHODS:
            images_src = os.path.join(OUTPUT_BASE_DIR, method, degradation)

            if not os.path.exists(images_src):
                print(f"WARNING: Missing folder, skipped: {images_src}")
                continue

            row = evaluate_dataset(
                model=model,
                degradation=degradation,
                method=method,
                images_src=images_src,
                labels_src=EVAL_LABELS_DIR
            )

            rows.append(row)

    if os.path.exists(TEMP_EVAL_DIR):
        shutil.rmtree(TEMP_EVAL_DIR)

    csv_path = os.path.join(RESULTS_DIR, "detection_results.csv")
    save_csv(rows, csv_path)

    print_summary_table(rows)

    print(f"\nCSV saved to: {csv_path}")
    print(f"Visual YOLO examples saved to: {VISUALS_DIR}")


if __name__ == "__main__":
    main()