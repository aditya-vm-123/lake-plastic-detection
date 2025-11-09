# validate_frcnn_yololike.py
"""
YOLO-like validator for Faster R-CNN checkpoints.
Produces:
- boxPR_curve.png, boxF1_curve.png, boxP_curve.png, boxR_curve.png
- confusion_matrix.png, confusion_matrix_normalised.png
- labels.png
- coco_summary.txt (COCO table)
- per_class_ap.csv
- results.csv (YOLO-ish columns; loss cols = NaN unless you log them)
"""

import os, json, argparse, math, time, itertools
from collections import defaultdict, Counter

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

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

def box_iou_xyxy(a, b):
    if len(a)==0 or len(b)==0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ixmin = np.maximum(a[:,None,0], b[None,:,0])
    iymin = np.maximum(a[:,None,1], b[None,:,1])
    ixmax = np.minimum(a[:,None,2], b[None,:,2])
    iymax = np.minimum(a[:,None,3], b[None,:,3])
    iw = np.clip(ixmax - ixmin, a_min=0, a_max=None)
    ih = np.clip(iymax - iymin, a_min=0, a_max=None)
    inter = iw * ih
    area_a = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union = area_a[:,None] + area_b[None,:] - inter
    return inter / np.clip(union, 1e-9, None)

def match_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thr=0.5):
    n_pred = len(pred_boxes)
    tp = np.zeros(n_pred, dtype=bool)
    pred_cls = pred_labels.copy()
    matched_gt_flags = np.zeros(len(gt_boxes), dtype=bool)

    classes = np.unique(np.concatenate([pred_labels, gt_labels])) if len(gt_labels)>0 or len(pred_labels)>0 else np.array([],dtype=int)
    for c in classes:
        p_idx = np.where(pred_labels==c)[0]
        g_idx = np.where(gt_labels==c)[0]
        if len(p_idx)==0 or len(g_idx)==0:
            continue
        iou = box_iou_xyxy(pred_boxes[p_idx], gt_boxes[g_idx])
        used_g = set()
        for k in np.argsort(-pred_scores[p_idx]):
            j = p_idx[k]
            best = -1; best_iou = 0.0
            for gi, g in enumerate(g_idx):
                if g in used_g: continue
                i = iou[k, gi]
                if i > best_iou:
                    best_iou = i; best = g
            if best >= 0 and best_iou >= iou_thr:
                tp[j] = True
                used_g.add(best)
                matched_gt_flags[np.where(g_idx==best)[0][0]] = True

    gt_cls_for_pred = np.full(n_pred, -1, dtype=int)
    if len(gt_boxes)>0 and n_pred>0:
        iou_all = box_iou_xyxy(pred_boxes, gt_boxes)
        best_gt = np.argmax(iou_all, axis=1) if iou_all.size>0 else np.full(n_pred, -1)
        ok = (tp==True)
        gt_idx = best_gt[ok]
        gt_cls_for_pred[ok] = gt_labels[gt_idx] if len(gt_idx)>0 else gt_cls_for_pred[ok]

    return tp, pred_cls, gt_cls_for_pred, matched_gt_flags

def plot_curve(x, y, xlabel, ylabel, title, path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(True)
    plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_confusion(cm, names, normalize, path):
    if normalize:
        cm = cm.astype(np.float64)
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(cm_sum, 1e-9))
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45, ha='right')
    plt.yticks(tick_marks, names)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def bar_labels(counts, names, path, title='Labels (val)'):
    plt.figure(figsize=(8,4))
    xs = np.arange(len(names))
    plt.bar(xs, counts)
    plt.xticks(xs, names, rotation=45, ha='right')
    plt.title(title); plt.grid(axis='y')
    plt.savefig(path, bbox_inches='tight'); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold for PR/F1/confusion (default 0.50)")
    ap.add_argument("--outdir", default="val_yolo_like")
    ap.add_argument("--score_min", type=float, default=0.0)
    ap.add_argument("--score_max", type=float, default=0.95)
    ap.add_argument("--score_steps", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = load_names(args.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco = json.load(open(args.val_json, "r", encoding="utf-8"))
    id2img = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    model = build_model(args.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    gt_label_counts = Counter([a["category_id"] for a in coco["annotations"]])
    gt_counts_list = [gt_label_counts.get(i+1, 0) for i in range(args.num_classes)]
    bar_labels(gt_counts_list, names, os.path.join(args.outdir, "labels.png"))

    score_thresholds = np.linspace(args.score_min, args.score_max, args.score_steps)
    precisions = []; recalls = []; f1s = []

    best_f1 = -1; best_thr = 0.25
    cache_preds = {}

    valid_ext = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    all_preds = {}
    for im in coco["images"]:
        fn = im["file_name"]
        p = os.path.join(args.val_images, fn)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).float().div(255.0).to(device)
        with torch.no_grad():
            out = model([t])[0]
        boxes = out.get("boxes", torch.empty(0,4)).cpu().numpy()
        scores = out.get("scores", torch.empty(0)).cpu().numpy()
        labels = out.get("labels", torch.empty(0, dtype=torch.long)).cpu().numpy()
        all_preds[im["id"]] = (boxes, scores, labels)

    gt_by_img = {}
    for iid, im in id2img.items():
        g = anns_by_img.get(iid, [])
        if len(g)==0:
            gt_by_img[iid] = (np.zeros((0,4)), np.zeros((0,), dtype=int))
            continue
        g_boxes = []
        g_labels = []
        for a in g:
            x,y,w,h = a["bbox"]
            g_boxes.append([x, y, x+w, y+h])
            g_labels.append(int(a["category_id"]))
        gt_by_img[iid] = (np.array(g_boxes, dtype=np.float32), np.array(g_labels, dtype=int))

    def micro_pr_at(thr):
        TP = 0; FP = 0; FN = 0
        for iid in id2img.keys():
            pred = all_preds.get(iid, (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)))
            p_boxes, p_scores, p_labels = pred
            keep = p_scores >= thr
            p_boxes = p_boxes[keep]; p_scores = p_scores[keep]; p_labels = p_labels[keep]
            g_boxes, g_labels = gt_by_img[iid]
            if len(p_boxes)==0 and len(g_boxes)==0:
                continue
            tp, pred_cls, gt_cls_for_pred, matched_gt_flags = match_detections(
                p_boxes, p_scores, p_labels, g_boxes, g_labels, iou_thr=args.iou
            )
            TP += int(tp.sum())
            FP += int((~tp).sum())
            FN += int((~matched_gt_flags).sum())
        P = TP / max(TP+FP, 1)     # precision
        R = TP / max(TP+FN, 1)     # recall
        F1 = (2*P*R)/max(P+R, 1e-12)
        return P, R, F1

    for thr in score_thresholds:
        P, R, F1 = micro_pr_at(thr)
        precisions.append(P); recalls.append(R); f1s.append(F1)
        if F1 > best_f1:
            best_f1 = F1; best_thr = float(thr)

    # Save curves
    plot_curve(score_thresholds, precisions, "Confidence threshold", "Precision", "boxP_curve", os.path.join(args.outdir, "boxP_curve.png"))
    plot_curve(score_thresholds, recalls,    "Confidence threshold", "Recall",    "boxR_curve", os.path.join(args.outdir, "boxR_curve.png"))
    plot_curve(score_thresholds, f1s,        "Confidence threshold", "F1",        "boxF1_curve", os.path.join(args.outdir, "boxF1_curve.png"))

    plt.figure()
    plt.plot(recalls, precisions)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("boxPR_curve (micro @ IoU=%.2f)" % args.iou)
    plt.grid(True); plt.savefig(os.path.join(args.outdir, "boxPR_curve.png"), bbox_inches="tight"); plt.close()

    y_true = []; y_pred = []
    for iid in id2img.keys():
        p_boxes, p_scores, p_labels = all_preds.get(iid, (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)))
        keep = p_scores >= best_thr
        p_boxes = p_boxes[keep]; p_scores = p_scores[keep]; p_labels = p_labels[keep]
        g_boxes, g_labels = gt_by_img[iid]
        tp, pred_cls, gt_cls_for_pred, matched_gt_flags = match_detections(
            p_boxes, p_scores, p_labels, g_boxes, g_labels, iou_thr=args.iou
        )
        m = np.where(tp)[0]
        if len(m)>0:
            y_true.extend(gt_cls_for_pred[m].tolist())
            y_pred.extend(pred_cls[m].tolist())

    if len(y_true)==0:
        cm = np.zeros((args.num_classes, args.num_classes), dtype=int)
    else:
        y_true_0 = [t-1 for t in y_true]
        y_pred_0 = [p-1 for p in y_pred]
        cm = confusion_matrix(y_true_0, y_pred_0, labels=list(range(args.num_classes)))

    plot_confusion(cm, names, normalize=False, path=os.path.join(args.outdir, "confusion_matrix.png"))
    plot_confusion(cm, names, normalize=True,  path=os.path.join(args.outdir, "confusion_matrix_normalised.png"))

    # COCO metrics (mAP50, mAP50-95)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cocoGt = COCO(args.val_json)
    cocoGt.dataset.setdefault("info", {})
    cocoGt.dataset.setdefault("licenses", [])
    cocoGt.createIndex()

    results = []
    for iid in id2img.keys():
        p_boxes, p_scores, p_labels = all_preds.get(iid, (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)))
        for (x1,y1,x2,y2), s, l in zip(p_boxes, p_scores, p_labels):
            results.append({
                "image_id": int(iid),
                "category_id": int(l),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(s)
            })
    with open(os.path.join(args.outdir, "coco_det_results.json"), "w") as f:
        json.dump(results, f)

    cocoDt = cocoGt.loadRes(os.path.join(args.outdir, "coco_det_results.json"))
    e = COCOeval(cocoGt, cocoDt, iouType='bbox')
    e.evaluate(); e.accumulate(); e.summarize()

    with open(os.path.join(args.outdir, "coco_summary.txt"), "w") as f:
        for i, name in enumerate([
            "AP@[0.50:0.95]","AP@0.50","AP@0.75","AP small","AP medium","AP large",
            "AR@1","AR@10","AR@100","AR small","AR medium","AR large"
        ]):
            f.write(f"{name}: {e.stats[i]:.3f}\n")

    precision = e.eval['precision']
    K = precision.shape[2]
    per_class_ap = []
    for k in range(K):
        p = precision[:,:,k,0,2]
        p = p[p>-1]
        ap = float(np.mean(p)) if p.size>0 else 0.0
        per_class_ap.append(ap)
    with open(os.path.join(args.outdir, "per_class_ap.csv"), "w") as f:
        f.write("class,AP\n")
        for i, apc in enumerate(per_class_ap):
            f.write(f"{names[i] if i<len(names) else i+1},{apc:.4f}\n")


    import math
    results_csv = os.path.join(args.outdir, "results.csv")
    with open(results_csv, "w") as f:
        f.write("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n")
        f.write(f"NA,NA,NaN,NaN,NaN,{precisions[np.argmax(f1s)]:.5f},{recalls[np.argmax(f1s)]:.5f},{e.stats[1]:.5f},{e.stats[0]:.5f},NaN,NaN,NaN,NaN,NaN,NaN\n")

    # Save best-F1 threshold
    with open(os.path.join(args.outdir, "best_f1.txt"), "w") as f:
        f.write(f"best_f1={best_f1:.4f} at threshold={best_thr:.3f}\n")

    print("\n✅ Saved outputs in:", args.outdir)
    print("   - boxPR_curve.png, boxF1_curve.png, boxP_curve.png, boxR_curve.png")
    print("   - confusion_matrix.png, confusion_matrix_normalised.png, labels.png")
    print("   - coco_summary.txt, per_class_ap.csv, results.csv, best_f1.txt")

if __name__ == "__main__":
    main()
