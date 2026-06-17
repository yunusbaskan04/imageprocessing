import cv2
import numpy as np
import os

INPUT_DIR = os.path.join("dataset_eval", "images")
OUTPUT_BASE_DIR = "dataset_degraded"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Reproducibility için sabit seed
RANDOM_SEED = 42


def apply_low_light(img, gamma=2.5):
    """
    Low-light simulation using gamma correction.
    gamma > 1 darkens the image.
    """
    table = np.array([
        ((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


def apply_fog(img, alpha=0.5, fog_intensity=200):
    """
    Synthetic fog simulation using alpha blending with a light-gray layer.
    """
    fog_layer = np.full(img.shape, fog_intensity, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - alpha, fog_layer, alpha, 0)


def apply_gaussian_noise(img, rng, mean=0, std=60):
    """
    Additive Gaussian noise simulation.
    """
    noise = rng.normal(mean, std, img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    low_light_dir = os.path.join(OUTPUT_BASE_DIR, "low_light")
    fog_dir = os.path.join(OUTPUT_BASE_DIR, "fog")
    noise_dir = os.path.join(OUTPUT_BASE_DIR, "noise")

    ensure_dir(low_light_dir)
    ensure_dir(fog_dir)
    ensure_dir(noise_dir)

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input folder not found: {INPUT_DIR}")
        print("First run prepare_eval_subset.py to create dataset_eval/images.")
        return

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not image_files:
        print(f"WARNING: No images found in {INPUT_DIR}")
        return

    print("=" * 80)
    print("Generating synthetic degradations")
    print("=" * 80)
    print(f"Input folder: {INPUT_DIR}")
    print(f"Total images: {len(image_files)}")
    print("Low-light: gamma = 2.5")
    print("Fog: alpha = 0.5, fog layer intensity = 200")
    print("Gaussian noise: mean = 0, std = 60")
    print("=" * 80)

    for count, filename in enumerate(image_files, 1):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"WARNING: Could not read image, skipped: {img_path}")
            continue

        low_light_img = apply_low_light(img, gamma=2.5)
        fog_img = apply_fog(img, alpha=0.5, fog_intensity=200)
        noise_img = apply_gaussian_noise(img, rng=rng, mean=0, std=60)

        cv2.imwrite(os.path.join(low_light_dir, filename), low_light_img)
        cv2.imwrite(os.path.join(fog_dir, filename), fog_img)
        cv2.imwrite(os.path.join(noise_dir, filename), noise_img)

        if count % 25 == 0 or count == len(image_files):
            print(f"Progress: {count}/{len(image_files)} completed.")

    print("=" * 80)
    print("Synthetic degradation generation completed.")
    print(f"Outputs saved under: {OUTPUT_BASE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()