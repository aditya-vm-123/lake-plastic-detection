from pathlib import Path
import os, io, json
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form
from datetime import datetime
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
print("STATIC_DIR:", STATIC_DIR)
CROWD_DIR = (ROOT / "crowd_sourced_images")
CROWD_DIR.mkdir(exist_ok=True)

# ---------- MODEL CONFIG (from .env or system env vars) ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/frcnn_final.pth")
print("MODEL_PATH:", MODEL_PATH)
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto")
NAMES_PATH = os.getenv("NAMES_PATH", "models/names.txt")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 7))  # fallback to 7

# ---------- Optional imports ----------
_ultra_ok = True
try:
    from ultralytics import YOLO
except Exception:
    _ultra_ok = False

_torch_ok = True
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms.functional as TF
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except Exception:
    _torch_ok = False
    FastRCNNPredictor = None

# ---------- App ----------
app = FastAPI(title="Plastic Detector (fresh)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Serve new static ONLY
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
@app.get("/", response_class=FileResponse)
def index():
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(404, f"index.html not found at {idx}")
    return FileResponse(str(idx))

# ---------- Model loader ----------
_model = None
_model_kind = None
_class_names: List[str] = []

def _decide_backend():
    if MODEL_BACKEND in ("yolo", "frcnn"):
        return MODEL_BACKEND
    return "frcnn" if MODEL_PATH.lower().endswith(".pth") else "yolo"

def _load_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _build_frcnn(num_classes_plus_bg: int):
    # try v2 then fallback
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as make
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as W
            model = make(weights=W.COCO_V1)
        except Exception:
            model = make()
    except Exception:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn as make
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as W
            model = make(weights=W.COCO_V1)
        except Exception:
            model = make()

    in_ch = model.roi_heads.box_predictor.cls_score.in_features
    if FastRCNNPredictor is not None:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes_plus_bg)
    else:
        class SimpleHead(nn.Module):
            def __init__(self, in_ch, n):
                super().__init__()
                self.cls = nn.Linear(in_ch, n)
                self.reg = nn.Linear(in_ch, n * 4)
            def forward(self, x):
                if isinstance(x, tuple): x = x[0]
                return self.cls(x), self.reg(x)
        model.roi_heads.box_predictor = SimpleHead(in_ch, num_classes_plus_bg)
    return model

def get_model():
    global _model, _model_kind, _class_names
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    kind = _decide_backend()
    _model_kind = kind

    if kind == "yolo":
        if not _ultra_ok:
            raise RuntimeError("Ultralytics not installed")
        print(f"Loading YOLO: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
        return _model

    # FRCNN
    if not _torch_ok:
        raise RuntimeError("PyTorch/TorchVision not installed")
    print(f"Loading FRCNN: {MODEL_PATH}")
    _class_names = _load_names(NAMES_PATH)
    if len(_class_names) != NUM_CLASSES:
        print(f"[WARN] names.txt has {len(_class_names)} lines; NUM_CLASSES={NUM_CLASSES}")

    model = _build_frcnn(NUM_CLASSES + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(MODEL_PATH, map_location=device)
    state = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    _model = (model, device)
    return _model

# ---------- Inference helpers ----------
def infer_yolo(model, img_rgb: np.ndarray, conf: float, imgsz: int = 768):
    res = model.predict(img_rgb, imgsz=imgsz, conf=conf, verbose=False)[0]
    dets = []
    if getattr(res, "boxes", None) is None:
        return dets
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    names = getattr(model, "names", {})
    for i, c in enumerate(clss):
        name = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
        x1,y1,x2,y2 = xyxy[i].tolist()
        dets.append({"class_id": int(c), "class_name": name, "conf": float(confs[i]), "box": [x1,y1,x2,y2]})
    return dets

def infer_frcnn(mod_dev, img_rgb: np.ndarray, score_thresh: float, names: List[str]):
    import torch
    model, device = mod_dev
    t = TF.to_tensor(Image.fromarray(img_rgb)).to(device)
    with torch.no_grad():
        out = model([t])[0]
    boxes = out.get("boxes"); scores = out.get("scores"); labels = out.get("labels")
    if boxes is None: return []
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype(int)
    dets = []
    for (x1,y1,x2,y2), sc, lab in zip(boxes, scores, labels):
        if sc < score_thresh: continue
        cid = int(lab) - 1  # shift to 0-based
        cname = names[cid] if 0 <= cid < len(names) else str(lab)
        dets.append({"class_id": cid, "class_name": cname, "conf": float(sc), "box": [float(x1),float(y1),float(x2),float(y2)]})
    return dets

# ---------- API ----------
@app.post("/predict")
async def predict(image: UploadFile = File(...), conf: float = 0.30):
    data = await image.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(pil)
    except Exception as e:
        return JSONResponse({"error": f"Invalid image: {e}"}, status_code=status.HTTP_400_BAD_REQUEST)
    try:
        m = get_model()
        kind = _model_kind or _decide_backend()
        if kind == "yolo":
            dets = infer_yolo(m, arr, conf=float(conf))
        else:
            dets = infer_frcnn(m, arr, score_thresh=float(conf), names=_class_names or _load_names(NAMES_PATH))
    except Exception as e:
        return JSONResponse({"error": f"Inference error: {e}"}, status_code=500)

    counts: Dict[str,int] = {}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return {"detections": dets, "counts": counts, "image_name": image.filename}


@app.post("/contribute")
async def contribute(
    image: UploadFile = File(...),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
):
    """
    Save a crowd-sourced image and its location with a sequential stem:
      0001.jpg (or .png/.webp) and 0001.txt (with lat,lng & metadata)
    Location is mandatory: prefer form lat/lng; if missing, try EXIF.
    """
    data = await image.read()
    # If lat/lng not provided, try EXIF
    lat2, lng2 = (lat, lng)
    if lat2 is None or lng2 is None:
        exif_lat, exif_lng = _gps_from_exif_bytes(data)
        if lat2 is None: lat2 = exif_lat
        if lng2 is None: lng2 = exif_lng

    if lat2 is None or lng2 is None:
        raise HTTPException(status_code=400, detail="Location required (provide lat/lng or image with EXIF GPS).")

    # Determine extension (preserve original)
    ext = (Path(image.filename).suffix or ".jpg").lower().lstrip(".")
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        ext = "jpg"  # fallback

    seq = _next_seq_id()
    stem = f"{seq:04d}"
    img_path = CROWD_DIR / f"{stem}.{ext}"
    txt_path = CROWD_DIR / f"{stem}.txt"

    # Save image bytes
    with open(img_path, "wb") as f:
        f.write(data)

    # Save txt with location + metadata
    meta = {
        "lat": float(lat2),
        "lng": float(lng2),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "original_filename": image.filename,
        "exif_used": (lat is None or lng is None),
    }
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2))

    return {
        "status": "ok",
        "id": stem,
        "image": str(img_path),
        "info": str(txt_path),
        "lat": meta["lat"],
        "lng": meta["lng"]
    }


@app.on_event("startup")
def _warm():
    try:
        get_model()
        print("Model preloaded.")
    except Exception as e:
        print("Model preload failed:", e)

@app.get("/health")
def health():
    return {"status": "ok"}


def _gps_from_exif_bytes(img_bytes: bytes):
    """Try reading GPS from EXIF (if piexif available). Returns (lat, lng) or (None, None)."""
    try:
        import piexif  # optional dependency
        from PIL import Image
        im = Image.open(io.BytesIO(img_bytes))
        exif_dict = piexif.load(im.info.get("exif", b""))
        gps = exif_dict.get("GPS", {})
        if not gps:
            return None, None

        def _to_deg(values, ref):
            # values like ((num, den), (num, den), (num, den))
            def frac(x): return x[0] / x[1] if x and x[1] else 0
            d = frac(values[0]); m = frac(values[1]); s = frac(values[2])
            val = d + (m / 60.0) + (s / 3600.0)
            if ref in [b'S', b'W']:
                val = -val
            return val

        lat = lng = None
        if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLatitudeRef in gps:
            lat = _to_deg(gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef])
        if piexif.GPSIFD.GPSLongitude in gps and piexif.GPSIFD.GPSLongitudeRef in gps:
            lng = _to_deg(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
        return lat, lng
    except Exception:
        return None, None


def _next_seq_id():
    """Scan CROWD_DIR for ####.ext, return next integer (1-based)."""
    import re
    pat = re.compile(r"^(\d{4})\.(jpg|jpeg|png|webp)$", re.I)
    max_id = 0
    for p in CROWD_DIR.iterdir():
        m = pat.match(p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1
