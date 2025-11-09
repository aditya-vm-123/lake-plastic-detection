# convert_splits_to_coco.py
# Convert YOLOv8 txt labels (class cx cy w h normalized) to COCO DETECTION JSON
# for all splits inside a split folder (expects train/val[/test]/images+labels).
import os, json, argparse
from pathlib import Path
from PIL import Image

def load_names(p):
    with open(p, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def one_split_to_coco(images_dir, labels_dir, out_json, categories):
    images_dir = Path(images_dir); labels_dir = Path(labels_dir)
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    images, annotations = [], []
    img_id, ann_id = 1, 1
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

    for img_path in files:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        lab_path = labels_dir / (img_path.stem + ".txt")
        if lab_path.exists():
            with open(lab_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c = int(float(parts[0]))
                    cx = float(parts[1]) * w
                    cy = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h

                    x1 = int(round(cx - bw/2.0))
                    y1 = int(round(cy - bh/2.0))
                    bw_i = int(round(bw))
                    bh_i = int(round(bh))

                    if bw_i <= 0 or bh_i <= 0:
                        continue
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x1 + bw_i > w: bw_i = w - x1
                    if y1 + bh_i > h: bh_i = h - y1
                    if bw_i <= 0 or bh_i <= 0:
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": c + 1,
                        "bbox": [x1, y1, bw_i, bh_i],
                        "area": int(bw_i * bh_i),
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"Wrote {out_json}  images={len(images)}  anns={len(annotations)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", required=True, help="e.g. split_raw or split_aug")
    ap.add_argument("--names", required=True, help="names.txt (7 lines)")
    ap.add_argument("--out_prefix", default=None, help="prefix for output files; defaults to folder name")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    names = load_names(args.names)
    categories = [{"id": i+1, "name": names[i]} for i in range(len(names))]

    if args.out_prefix:
        prefix = args.out_prefix
    else:
        prefix = split_root.name

    # subfolders
    candidates = [("train", "train"), ("val","val"), ("test","test")]
    for sub, tag in candidates:
        img_dir = split_root / sub / "images"
        lab_dir = split_root / sub / "labels"
        if img_dir.exists() and lab_dir.exists():
            out_json = f"{prefix}_{tag}_det.json"
            one_split_to_coco(img_dir, lab_dir, out_json, categories)
        else:
            print(f"Skip: {img_dir} or {lab_dir} missing")

if __name__ == "__main__":
    main()
