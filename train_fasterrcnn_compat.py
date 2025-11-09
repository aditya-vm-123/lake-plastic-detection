# train_fasterrcnn_compat.py
import os, argparse, datetime, torch
from torch.utils.data import DataLoader
from coco_det_dataset import CocoDetDataset, collate_fn

HAS_V2 = True
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as frcnn_fn
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as W_V2
        DEFAULT_WEIGHTS = dict(weights=W_V2.COCO_V1)
    except Exception:
        DEFAULT_WEIGHTS = dict(weights="COCO_V1")
except Exception:
    HAS_V2 = False
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as frcnn_fn
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as W_FPN
        DEFAULT_WEIGHTS = dict(weights=W_FPN.COCO_V1)
    except Exception:
        DEFAULT_WEIGHTS = dict()
        pass

try:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except Exception:
    FastRCNNPredictor = None

import torch.nn as nn
class SimpleFasterRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def get_model(num_classes: int):
    kw = {}
    if DEFAULT_WEIGHTS:
        kw.update(DEFAULT_WEIGHTS)
    try:
        model = frcnn_fn(**kw)
    except TypeError:
        model = frcnn_fn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    if FastRCNNPredictor is not None:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    else:
        model.roi_heads.box_predictor = SimpleFasterRCNNPredictor(in_features, num_classes + 1)
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Using backbone:", "R50-FPNv2" if HAS_V2 else "R50-FPN")

    os.makedirs(args.outdir, exist_ok=True)

    train_ds = CocoDetDataset(args.train_images, args.train_json)
    val_ds = CocoDetDataset(args.val_images, args.val_json) if args.val_json and os.path.exists(args.val_json) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn) if val_ds else None

    model = get_model(args.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        t0 = datetime.datetime.now()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()); n += 1

        lr_sched.step()
        dt = datetime.datetime.now() - t0
        avg_loss = loss_sum / max(1, n)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_train_loss={avg_loss:.4f}  time={dt}")

        ckpt = os.path.join(args.outdir, f"frcnn_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt)
        print("Saved", ckpt)

        if val_loader and args.quick_val > 0:
            model.train()
            vloss, m = 0.0, 0
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                with torch.set_grad_enabled(False):
                    ldict = model(images, targets)
                    vloss += float(sum(ldict.values()).item()); m += 1
                if m >= args.quick_val: break
            print(f"Quick val loss (first {m} imgs): {vloss/max(1,m):.4f}")

    final = os.path.join(args.outdir, "frcnn_final.pth")
    torch.save(model.state_dict(), final)
    print("Saved", final)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_images", default=None)
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2)    # set 1 if OOM
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--step_size", type=int, default=5)
    ap.add_argument("--outdir", default="frcnn_output")
    ap.add_argument("--quick_val", type=int, default=200)
    args = ap.parse_args()
    main(args)
