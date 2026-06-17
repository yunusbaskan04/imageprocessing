import os
import cv2
import matplotlib.pyplot as plt

IMAGES_DIR = "dataset_eval/images"
LABELS_DIR = "dataset_eval/labels"
OUT_PATH = "results/figures/sample_contact_sheet.png"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_labelled_images():
    images = []
    for f in os.listdir(IMAGES_DIR):
        if not f.lower().endswith(VALID_EXTS):
            continue

        stem, _ = os.path.splitext(f)
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if lines:
            images.append(f)

    return images

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    images = list_labelled_images()[:40]

    cols = 5
    rows = 8

    fig, axes = plt.subplots(rows, cols, figsize=(15, 20))
    axes = axes.flatten()

    for ax, img_name in zip(axes, images):
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

        ax.set_title(img_name[:25], fontsize=7)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved contact sheet: {OUT_PATH}")

if __name__ == "__main__":
    main()