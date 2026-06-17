import argparse
import os
import shutil
import cv2
import numpy as np

INPUT_BASE_DIR = "dataset_degraded"
OUTPUT_BASE_DIR = "outputs"

DEGRADATIONS = ["low_light", "fog", "noise"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

METHODS = [
    "degraded",
    "global_he",
    "clahe_only",
    "proposed",
    "morphology_ablation"
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def apply_global_he(img):
    """
    Global Histogram Equalization baseline.
    Histogram equalization is applied to the luminance channel only
    to reduce color distortion.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_eq = cv2.equalizeHist(y)

    merged = cv2.merge((y_eq, cr, cb))
    out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    return out


def apply_clahe_only(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE-only baseline.
    CLAHE is applied only to the L channel in LAB color space.
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_clahe = clahe.apply(l)
    merged_lab = cv2.merge((l_clahe, a, b))
    out = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return out


def apply_proposed(img):
    """
    Proposed pipeline:
    LAB color conversion + CLAHE on L channel + Bilateral Filter.
    """
    clahe_img = apply_clahe_only(
        img,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    )

    filtered_img = cv2.bilateralFilter(
        clahe_img,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    return filtered_img


def apply_morphology_ablation(img):
    """
    Ablation variant:
    Proposed pipeline followed by morphological closing.
    This is used to test whether morphology helps or oversmooths small UAV details.
    """
    proposed_img = apply_proposed(img)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    closed_img = cv2.morphologyEx(
        proposed_img,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return closed_img


def enhance_by_method(img, method):
    if method == "degraded":
        return img
    if method == "global_he":
        return apply_global_he(img)
    if method == "clahe_only":
        return apply_clahe_only(img)
    if method == "proposed":
        return apply_proposed(img)
    if method == "morphology_ablation":
        return apply_morphology_ablation(img)

    raise ValueError(f"Unknown method: {method}")


def process_method(method):
    print("\n" + "=" * 80)
    print(f"Processing method: {method}")
    print("=" * 80)

    for degradation in DEGRADATIONS:
        input_dir = os.path.join(INPUT_BASE_DIR, degradation)
        output_dir = os.path.join(OUTPUT_BASE_DIR, method, degradation)

        ensure_dir(output_dir)

        if not os.path.exists(input_dir):
            print(f"WARNING: Input folder not found, skipped: {input_dir}")
            continue

        image_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        if not image_files:
            print(f"WARNING: No images found in {input_dir}")
            continue

        print(f"[{method} / {degradation}] {len(image_files)} images")

        for count, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if method == "degraded":
                shutil.copy2(input_path, output_path)
                continue

            img = cv2.imread(input_path)

            if img is None:
                print(f"WARNING: Could not read image, skipped: {input_path}")
                continue

            out_img = enhance_by_method(img, method)
            cv2.imwrite(output_path, out_img)

            if count % 50 == 0 or count == len(image_files):
                print(f"  Progress: {count}/{len(image_files)}")

    print(f"Method completed: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline and proposed enhancement outputs."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all"] + METHODS,
        help="Enhancement method to run."
    )

    args = parser.parse_args()

    if not os.path.exists(INPUT_BASE_DIR):
        print(f"ERROR: Input folder not found: {INPUT_BASE_DIR}")
        print("First run generate_degradations.py.")
        return

    if args.method == "all":
        methods_to_run = METHODS
    else:
        methods_to_run = [args.method]

    print("=" * 80)
    print("Enhancement baseline generation")
    print("=" * 80)
    print(f"Input base folder: {INPUT_BASE_DIR}")
    print(f"Output base folder: {OUTPUT_BASE_DIR}")
    print(f"Methods: {methods_to_run}")
    print("=" * 80)

    for method in methods_to_run:
        process_method(method)

    print("\n" + "=" * 80)
    print("All enhancement outputs are ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()