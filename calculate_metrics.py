import os
import csv
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ORIGINAL_DIR = os.path.join("dataset_eval", "images")
OUTPUT_BASE_DIR = "outputs"
RESULTS_DIR = "results"

DEGRADATIONS = ["low_light", "fog", "noise"]

METHODS = [
    "degraded",
    "global_he",
    "clahe_only",
    "proposed",
    "morphology_ablation"
]

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


def calculate_metrics_for_directory(original_dir, target_dir):
    psnr_values = []
    ssim_values = []
    missing_files = 0
    unreadable_files = 0

    original_files = list_images(original_dir)

    for filename in original_files:
        original_path = os.path.join(original_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if not os.path.exists(target_path):
            missing_files += 1
            continue

        original_img = cv2.imread(original_path)
        target_img = cv2.imread(target_path)

        if original_img is None or target_img is None:
            unreadable_files += 1
            continue

        if original_img.shape != target_img.shape:
            target_img = cv2.resize(
                target_img,
                (original_img.shape[1], original_img.shape[0])
            )

        psnr_value = psnr(original_img, target_img, data_range=255)
        ssim_value = ssim(
            original_img,
            target_img,
            channel_axis=-1,
            data_range=255
        )

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    if not psnr_values:
        return {
            "count": 0,
            "psnr": 0.0,
            "ssim": 0.0,
            "missing": missing_files,
            "unreadable": unreadable_files
        }

    return {
        "count": len(psnr_values),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "missing": missing_files,
        "unreadable": unreadable_files
    }


def save_csv(rows, csv_path):
    fieldnames = [
        "degradation",
        "method",
        "image_count",
        "psnr",
        "ssim",
        "missing_files",
        "unreadable_files"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_table(rows):
    print("\n" + "=" * 100)
    print("IMAGE QUALITY RESULTS: PSNR / SSIM")
    print("=" * 100)
    print(
        f"| {'Degradation':<12} | {'Method':<20} | {'Images':<6} | "
        f"{'PSNR':<10} | {'SSIM':<10} | {'Missing':<8} | {'Unreadable':<10} |"
    )
    print("-" * 100)

    for row in rows:
        print(
            f"| {row['degradation']:<12} | "
            f"{row['method']:<20} | "
            f"{row['image_count']:<6} | "
            f"{row['psnr']:<10.4f} | "
            f"{row['ssim']:<10.4f} | "
            f"{row['missing_files']:<8} | "
            f"{row['unreadable_files']:<10} |"
        )

    print("=" * 100)


def main():
    ensure_dir(RESULTS_DIR)

    if not os.path.exists(ORIGINAL_DIR):
        print(f"ERROR: Original image directory not found: {ORIGINAL_DIR}")
        print("First run prepare_eval_subset.py.")
        return

    rows = []

    print("=" * 100)
    print("Calculating PSNR and SSIM for all methods")
    print("=" * 100)
    print(f"Original folder: {ORIGINAL_DIR}")
    print(f"Output base folder: {OUTPUT_BASE_DIR}")
    print("=" * 100)

    for degradation in DEGRADATIONS:
        for method in METHODS:
            target_dir = os.path.join(OUTPUT_BASE_DIR, method, degradation)

            if not os.path.exists(target_dir):
                print(f"WARNING: Target folder not found, skipped: {target_dir}")
                continue

            result = calculate_metrics_for_directory(
                ORIGINAL_DIR,
                target_dir
            )

            row = {
                "degradation": degradation,
                "method": method,
                "image_count": result["count"],
                "psnr": result["psnr"],
                "ssim": result["ssim"],
                "missing_files": result["missing"],
                "unreadable_files": result["unreadable"]
            }

            rows.append(row)

    csv_path = os.path.join(RESULTS_DIR, "image_quality_results.csv")
    save_csv(rows, csv_path)

    print_table(rows)

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()