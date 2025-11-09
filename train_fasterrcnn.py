# train_fasterrcnn.py
import os, argparse, datetime
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
from coco_det_dataset import CocoDetDataset, collate_fn

def get_model(num_classes: int):
    # Faster R-CNN with ResNet50-FPNv2 backbone (good default)
    model = fasterrcnn_resnet50_fpn_v2(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # new predictor: num_classes includes background -> so use num_classes+1 if your labels start at 1
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.outdir, exist_ok=True)

    # datasets
    train_ds = CocoDetDataset(args.train_images, args.train_json)
    val_ds = CocoDetDataset(args.val_images, args.val_json) if args.val_json and os.path.exists(args.val_json) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn) if val_ds else None

    num_classes = args.num_classes  # 7 for your case
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    best_train_loss = 1e9
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

            loss_sum += loss.item()
            n += 1

        lr_sched.step()
        dt = datetime.datetime.now() - t0
        avg_loss = loss_sum / max(1, n)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_train_loss={avg_loss:.4f}  time={dt}")

        # save checkpoint
        ckpt = os.path.join(args.outdir, f"frcnn_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt)
        print("Saved", ckpt)

        # quick val (optional): compute simple classification+box regression loss on first N val images
        if val_loader and args.quick_val>0:
            model.train()  # faster-rcnn computes losses only in train mode with targets
            val_loss_sum, m = 0.0, 0
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                with torch.set_grad_enabled(False):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                val_loss_sum += float(loss.item())
                m += 1
                if m >= args.quick_val:
                    break
            print(f"Quick val loss (first {m} images): {val_loss_sum/max(1,m):.4f}")

    # final save
    final = os.path.join(args.outdir, "frcnn_final.pth")
    torch.save(model.state_dict(), final)
    print("Saved", final)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_images", default=None)
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--num_classes", type=int, required=True, help="number of object classes (7 for your dataset)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2)  # set 1 if OOM
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--step_size", type=int, default=5)
    ap.add_argument("--outdir", default="frcnn_output")
    ap.add_argument("--quick_val", type=int, default=200, help="evaluate first N val images by loss; 0 to disable")
    args = ap.parse_args()
    main(args)
