#!/usr/bin/env python3
"""
augment_yolo_dataset.py

Produces an augmented YOLO-format dataset (images + .txt labels).
- Clips bbox coords to [0,1], removes tiny boxes.
- Uses "images-per-class" as the balancing target (avoids exploding majority boxes).
- Copies originals into output and appends augmented files.
- Configurable parameters at top or via CLI.

Example:
    python augment_yolo_dataset.py
"""

from pathlib import Path
import argparse
import random
import shutil
import sys
from collections import defaultdict, Counter
import cv2
import albumentations as A
import numpy as np

# ---------------- CONFIG ----------------
DEFAULT_BASE = Path(r"C:\Users\adity\LakeProject\consolidated_dataset")
DEFAULT_SRC_IMAGES = DEFAULT_BASE / "raw_images" / "images"
DEFAULT_SRC_LABELS = DEFAULT_BASE / "raw_images" / "labels"
DEFAULT_OUT_ROOT = DEFAULT_BASE / "aug_images"
BASELINE_AUG_PER_IMAGE = 1
TARGET_IMAGES_PER_CLASS = 200
MAX_EXTRA_AUGS_PER_IMAGE = 2
MIN_BOX_WH = 0.005
RANDOM_SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_images", type=Path, default=DEFAULT_SRC_IMAGES)
    p.add_argument("--src_labels", type=Path, default=DEFAULT_SRC_LABELS)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--base_aug_per_image", type=int, default=BASELINE_AUG_PER_IMAGE)
    p.add_argument("--target_images_per_class", type=int, default=TARGET_IMAGES_PER_CLASS)
    p.add_argument("--max_extra_augs_per_image", type=int, default=MAX_EXTRA_AUGS_PER_IMAGE)
    p.add_argument("--min_box_wh", type=float, default=MIN_BOX_WH)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--no_copy_originals", action="store_true", help="Don't copy originals into out folder (default: copy)")
    return p.parse_args()

def read_yolo_labels(txt_path: Path):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                cls = int(parts[0])
                coords = list(map(float, parts[1:5]))
                if len(coords) != 4:
                    continue
                boxes.append((cls, coords[0], coords[1], coords[2], coords[3]))
            except Exception:
                continue
    return boxes

def write_yolo_label(txt_path: Path, boxes):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for b in boxes:
            cls = int(b[0])
            coords = b[1:]
            f.write(f"{cls} " + " ".join([f"{x:.6f}" for x in coords]) + "\n")

def sanitize_and_filter_bboxes(aug_bboxes, aug_labels, min_wh=0.005):
    clean = []
    for bbox, label in zip(aug_bboxes, aug_labels):
        x, y, w, h = bbox
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        nw = x2 - x1
        nh = y2 - y1
        if nw <= 0 or nh <= 0:
            continue
        if nw < min_wh or nh < min_wh:
            continue
        nx = x1 + nw / 2.0
        ny = y1 + nh / 2.0
        clean.append((int(label), nx, ny, nw, nh))
    return clean

def build_transform():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=10, p=0.6, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.12),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.12),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.12),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    return transform

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    SRC_IMAGES_DIR = args.src_images
    SRC_LABELS_DIR = args.src_labels
    OUT_ROOT = args.out_root
    OUT_IMAGES = OUT_ROOT / "images"
    OUT_LABELS = OUT_ROOT / "labels"

    print("Configuration:")
    print(" SRC_IMAGES_DIR:", SRC_IMAGES_DIR)
    print(" SRC_LABELS_DIR:", SRC_LABELS_DIR)
    print(" OUT_ROOT:", OUT_ROOT)
    print(" base_aug_per_image:", args.base_aug_per_image)
    print(" target_images_per_class:", args.target_images_per_class)
    print(" max_extra_augs_per_image:", args.max_extra_augs_per_image)
    print(" min_box_wh:", args.min_box_wh)
    print()

    if not SRC_IMAGES_DIR.exists() or not SRC_IMAGES_DIR.is_dir():
        print(f"ERROR: source images folder not found: {SRC_IMAGES_DIR}")
        sys.exit(1)
    if not SRC_LABELS_DIR.exists() or not SRC_LABELS_DIR.is_dir():
        print(f"WARNING: source labels folder not found: {SRC_LABELS_DIR} - continuing but labels may be missing")

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    # gather images
    image_paths = sorted([p for p in SRC_IMAGES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"Found {len(image_paths)} source images.")

    # load labels and compute per-image box lists
    image_boxes = {}
    class_counts = Counter()
    images_with_class = defaultdict(set)
    for img_path in image_paths:
        lbl = SRC_LABELS_DIR / (img_path.stem + ".txt")
        boxes = read_yolo_labels(lbl)
        image_boxes[img_path] = boxes
        for b in boxes:
            class_counts[b[0]] += 1
            images_with_class[b[0]].add(img_path.stem)

    print("Current class box counts:", dict(class_counts))
    print("Current images-per-class:", {k: len(v) for k,v in images_with_class.items()})
    print()

    if not args.no_copy_originals:
        print("Copying originals to output folder...")
        for img_path in image_paths:
            dest_img = OUT_IMAGES / img_path.name
            shutil.copy2(img_path, dest_img)
            lbl_src = SRC_LABELS_DIR / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, OUT_LABELS / (img_path.stem + ".txt"))
    else:
        print("Skipping copying originals (no_copy_originals=True).")

    transform = build_transform()

    augmented_images_per_class = defaultdict(int)

    print("Starting baseline augmentations...")
    aug_id = 0
    baseline_created = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print("Warning: unable to read image:", img_path)
            continue
        boxes = image_boxes.get(img_path, [])
        if not boxes:
            continue
        yolo_bboxes = [(b[1], b[2], b[3], b[4]) for b in boxes]
        class_labels = [int(b[0]) for b in boxes]
        for i in range(max(0, args.base_aug_per_image)):
            try:
                aug = transform(image=img, bboxes=yolo_bboxes, class_labels=class_labels)
            except Exception as e:
                continue
            aug_img = aug["image"]
            aug_bboxes = aug["bboxes"]
            aug_labels = aug["class_labels"]
            cleaned = sanitize_and_filter_bboxes(aug_bboxes, aug_labels, min_wh=args.min_box_wh)
            if not cleaned:
                continue
            new_name = f"{img_path.stem}_aug{aug_id}.jpg"
            out_img_path = OUT_IMAGES / new_name
            out_lbl_path = OUT_LABELS / (Path(new_name).stem + ".txt")
            cv2.imwrite(str(out_img_path), aug_img)
            write_yolo_label(out_lbl_path, cleaned)
            classes_present = set([c for (c, *_c) in cleaned])
            for cls in classes_present:
                augmented_images_per_class[cls] += 1
            aug_id += 1
            baseline_created += 1

    print(f"Baseline augmentations done. Created {baseline_created} augmented images.")
    print("Partial augmented images per class:", dict(augmented_images_per_class))
    print()

    print("Starting targeted augmentation to increase images-per-class up to", args.target_images_per_class)
    orig_images_per_class = {cls: len(stems) for cls, stems in images_with_class.items()}

    classes_all = set(orig_images_per_class.keys())

    under_classes = {cls for cls in classes_all
                     if (orig_images_per_class.get(cls, 0) + augmented_images_per_class.get(cls, 0)) < args.target_images_per_class}

    print("Initial under-represented classes (by images):", under_classes)

    total_extra_cap = len(image_paths) * args.max_extra_augs_per_image
    extra_created = 0

    img_list = list(image_paths)
    random.shuffle(img_list)

    def existing_augments_for_stem(stem):
        count = 0
        count += len(list(OUT_IMAGES.glob(f"{stem}_aug*.jpg")))
        count += len(list(OUT_IMAGES.glob(f"{stem}_taug*.jpg")))
        return count

    pass_num = 0
    while under_classes and extra_created < total_extra_cap:
        pass_num += 1
        made_in_pass = 0
        for img_path in img_list:
            if extra_created >= total_extra_cap:
                break
            boxes = image_boxes.get(img_path, [])
            if not boxes:
                continue
            classes_in_img = set([b[0] for b in boxes])
            if not (classes_in_img & under_classes):
                continue
            stem = img_path.stem
            if existing_augments_for_stem(stem) >= args.max_extra_augs_per_image:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                aug = transform(image=img, bboxes=[(b[1], b[2], b[3], b[4]) for b in boxes], class_labels=[int(b[0]) for b in boxes])
            except Exception:
                continue
            cleaned = sanitize_and_filter_bboxes(aug.get('bboxes', []), aug.get('class_labels', []), min_wh=args.min_box_wh)
            if not cleaned:
                continue
            new_name = f"{stem}_taug{extra_created}.jpg"
            out_img_path = OUT_IMAGES / new_name
            out_lbl_path = OUT_LABELS / (Path(new_name).stem + ".txt")
            cv2.imwrite(str(out_img_path), aug["image"])
            write_yolo_label(out_lbl_path, cleaned)
            extra_created += 1
            made_in_pass += 1
            classes_present = set([c for (c, *_c) in cleaned])
            for cls in classes_present:
                augmented_images_per_class[cls] += 1
            under_classes = {cls for cls in classes_all
                             if (orig_images_per_class.get(cls, 0) + augmented_images_per_class.get(cls, 0)) < args.target_images_per_class}
            if not under_classes:
                break
        if made_in_pass == 0:
            break

    print(f"Targeted augmentation done. Created {extra_created} targeted images.")
    print("Final approx. augmented images per class (added):", dict(augmented_images_per_class))
    print("Final approximate images per class (orig + added):")
    for cls in sorted(classes_all):
        orig_count = orig_images_per_class.get(cls, 0)
        added = augmented_images_per_class.get(cls, 0)
        print(f"  class {cls}: {orig_count} orig + {added} aug = {orig_count + added}")

    print()
    print("Augmented dataset saved to:", OUT_ROOT)
    print("Remember: run your split script on the new OUT_ROOT (aug_images) and update data.yaml before training.")

if __name__ == "__main__":
    main()
