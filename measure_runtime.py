import os
import time
import csv
import cv2
import numpy as np

from enhance_images import (
    apply_global_he,
    apply_clahe_only,
    apply_proposed,
    apply_morphology_ablation
)

INPUT_BASE_DIR = "dataset_degraded"
RESULTS_DIR = "results"

DEGRADATIONS = ["low_light", "fog", "noise"]

METHOD_FUNCTIONS = {
    "global_he": apply_global_he,
    "clahe_only": apply_clahe_only,
    "clahe_bilateral": apply_proposed,
    "proposed_final": apply_morphology_ablation
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    if not os.path.exists(folder):
        return []

    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]


def measure_method_runtime(degradation, method_name, method_func):
    input_dir = os.path.join(INPUT_BASE_DIR, degradation)
    image_files = list_images(input_dir)

    if not image_files:
        return None

    images = []

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is not None:
            images.append(img)

    if not images:
        return None

    # Warm-up: first few runs are ignored
    for img in images[:10]:
        _ = method_func(img)

    start = time.perf_counter()

    for img in images:
        _ = method_func(img)

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / len(images)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    return {
        "degradation": degradation,
        "method": method_name,
        "image_count": len(images),
        "total_time_sec": total_time,
        "avg_time_ms": avg_time * 1000,
        "fps": fps
    }


def save_csv(rows, csv_path):
    fieldnames = [
        "degradation",
        "method",
        "image_count",
        "total_time_sec",
        "avg_time_ms",
        "fps"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print("\n" + "=" * 90)
    print("PRE-PROCESSING RUNTIME RESULTS")
    print("=" * 90)
    print(
        f"| {'Degradation':<12} | {'Method':<18} | {'Images':<6} | "
        f"{'Avg Time (ms)':<15} | {'FPS':<10} |"
    )
    print("-" * 90)

    for row in rows:
        print(
            f"| {row['degradation']:<12} | "
            f"{row['method']:<18} | "
            f"{row['image_count']:<6} | "
            f"{row['avg_time_ms']:<15.4f} | "
            f"{row['fps']:<10.2f} |"
        )

    print("=" * 90)


def main():
    ensure_dir(RESULTS_DIR)

    rows = []

    print("=" * 90)
    print("Measuring pre-processing runtime")
    print("=" * 90)

    for degradation in DEGRADATIONS:
        for method_name, method_func in METHOD_FUNCTIONS.items():
            result = measure_method_runtime(
                degradation=degradation,
                method_name=method_name,
                method_func=method_func
            )

            if result is not None:
                rows.append(result)

    csv_path = os.path.join(RESULTS_DIR, "runtime_results.csv")
    save_csv(rows, csv_path)
    print_table(rows)

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()