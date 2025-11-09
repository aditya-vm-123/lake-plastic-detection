# predict_test_raw.py (fixed)
import os, cv2, torch, argparse
import numpy as np

def load_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

# model builder
HAS_V2 = True
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as frcnn_fn
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as W_V2
        DEFAULT_WEIGHTS = dict(weights=W_V2.COCO_V1)
    except Exception:
        DEFAULT_WEIGHTS = {}
except Exception:
    HAS_V2 = False
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as frcnn_fn
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as W_FPN
        DEFAULT_WEIGHTS = dict(weights=W_FPN.COCO_V1)
    except Exception:
        DEFAULT_WEIGHTS = {}

try:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except Exception:
    FastRCNNPredictor = None

import torch.nn as nn
class SimplePredictor(nn.Module):
    def __init__(self, in_ch, classes):
        super().__init__()
        self.cls = nn.Linear(in_ch, classes)
        self.reg = nn.Linear(in_ch, classes * 4)
    def forward(self, x):
        if isinstance(x, tuple): x = x[0]
        return self.cls(x), self.reg(x)

def build_model(num_classes):
    kw = {}; kw.update(DEFAULT_WEIGHTS)
    try:
        model = frcnn_fn(**kw)
    except TypeError:
        model = frcnn_fn(pretrained=True)
    in_ch = model.roi_heads.box_predictor.cls_score.in_features
    if FastRCNNPredictor:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes+1)
    else:
        model.roi_heads.box_predictor = SimplePredictor(in_ch, num_classes+1)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--outdir", default="frcnn_vis_raw")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--score_thresh", type=float, default=0.30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    class_names = load_names(args.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(args.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    det_rows = ["file,x1,y1,x2,y2,score,label_id,label_name"]
    valid_ext = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

    for fn in sorted(os.listdir(args.images)):
        if not fn.lower().endswith(valid_ext): continue
        path = os.path.join(args.images, fn)

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("Skip unreadable:", fn); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0).to(device)

        with torch.no_grad():
            out = model([t])[0]

        vis = img_bgr.copy()
        boxes = out.get("boxes", torch.empty(0,4)).cpu().numpy()
        scores = out.get("scores", torch.empty(0)).cpu().numpy()
        labels = out.get("labels", torch.empty(0, dtype=torch.long)).cpu().numpy()

        for (x1,y1,x2,y2), sc, lab in zip(boxes, scores, labels):
            if sc < args.score_thresh: 
                continue
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            name = class_names[lab-1] if 1 <= lab <= len(class_names) else str(int(lab))
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{name} {sc:.2f}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            det_rows.append(f"{fn},{x1},{y1},{x2},{y2},{sc:.4f},{int(lab)},{name}")

        cv2.imwrite(os.path.join(args.outdir, fn), vis)

    csv_path = os.path.join(args.outdir, "detections.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(det_rows))
    print(f"Saved visualizations to: {args.outdir}")
    print(f"Saved detections CSV to: {csv_path}")

if __name__ == "__main__":
    main()
