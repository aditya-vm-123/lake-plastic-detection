import os
import random
import shutil
from pathlib import Path

# === CONFIG ===
RAW_DIR = Path("raw_images")
OUT_DIR = Path("split_raw")
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

# split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# output directories
for split in ["train", "val", "test"]:
    (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# collect paths
image_dir = RAW_DIR / "images"
label_dir = RAW_DIR / "labels"
images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

random.seed(42)
random.shuffle(images)

n_total = len(images)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)
n_test = n_total - n_train - n_val

splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:]
}

print(f"Total images: {n_total}")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# copy images + labels
for split_name, image_paths in splits.items():
    for img_path in image_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        dst_img = OUT_DIR / split_name / "images" / img_path.name
        dst_label = OUT_DIR / split_name / "labels" / label_path.name

        shutil.copy2(img_path, dst_img)
        if label_path.exists():
            shutil.copy2(label_path, dst_label)
        else:
            print(f"Missing label for {img_path.name}, skipped label copy")

print("\nDone splitting dataset!")
print(f"Saved to: {OUT_DIR.resolve()}")
