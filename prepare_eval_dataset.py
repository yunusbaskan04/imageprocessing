import os
import shutil

DATASET_DIR = "dataset_original"
OUTPUT_DIR = "dataset_eval"
SPLITS_TO_COMBINE = ["valid", "test"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def copy_split(split_name, output_images_dir, output_labels_dir):
    images_dir = os.path.join(DATASET_DIR, split_name, "images")
    labels_dir = os.path.join(DATASET_DIR, split_name, "labels")

    if not os.path.exists(images_dir):
        print(f"Missing images directory: {images_dir}")
        return 0, 0

    if not os.path.exists(labels_dir):
        print(f"Missing labels directory: {labels_dir}")
        return 0, 0

    image_count = 0
    label_count = 0

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(IMAGE_EXTS):
            src_img = os.path.join(images_dir, filename)

            # valid ve test içinde aynı dosya adı olursa çakışmasın diye prefix ekliyoruz
            new_img_name = f"{split_name}_{filename}"
            dst_img = os.path.join(output_images_dir, new_img_name)
            shutil.copy2(src_img, dst_img)
            image_count += 1

            stem, _ = os.path.splitext(filename)
            label_name = stem + ".txt"
            src_label = os.path.join(labels_dir, label_name)

            new_label_name = f"{split_name}_{stem}.txt"
            dst_label = os.path.join(output_labels_dir, new_label_name)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
                label_count += 1
            else:
                # Etiket yoksa boş label oluşturuyoruz ki YOLO formatı bozulmasın
                open(dst_label, "w", encoding="utf-8").close()

    return image_count, label_count

def main():
    output_images_dir = os.path.join(OUTPUT_DIR, "images")
    output_labels_dir = os.path.join(OUTPUT_DIR, "labels")

    ensure_clean_dir(OUTPUT_DIR)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    total_images = 0
    total_labels = 0

    print("=" * 70)
    print("Preparing combined evaluation subset: valid + test")
    print("=" * 70)

    for split in SPLITS_TO_COMBINE:
        image_count, label_count = copy_split(split, output_images_dir, output_labels_dir)
        total_images += image_count
        total_labels += label_count

        print(f"{split}: copied {image_count} images and {label_count} labels")

    print("=" * 70)
    print(f"Total copied images: {total_images}")
    print(f"Total copied labels: {total_labels}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()