# Lake Plastic Detection (FastAPI Web App)

A local prototype web app for detecting floating plastic in lake images using object detection models (YOLOv8 / YOLOv11 / Faster R-CNN). Users can upload images, view detections (bounding boxes + per-class counts), and optionally save crowd-sourced images with location metadata.

---

## Current status & important note about model files

Two large Faster R-CNN model checkpoints (raw/aug) exceed GitHub's 100 MB file limit and are therefore **not included** in this repository. The required Faster R-CNN model files are hosted in this Google Drive folder:

**Google Drive (models)**  
> https://drive.google.com/drive/folders/15SzBP0jWw9KYruj8iMVxtKCvNLVlT3sO?usp=drive_link

You must download the model files from that Drive folder and place them in the repository `models/` directory before running the app locally.

> Files to download (example names used in the code):
> - `frcnn_raw_final.pth`  (Faster R-CNN trained on raw dataset)
> - `frcnn_aug_final.pth`  (Faster R-CNN trained on augmented dataset)

Small model checkpoints used for other experiments (e.g., YOLO variants) are already present in the `models/` folder in the repo.

---

## Prerequisites

- Python 3.9+ (3.11 tested)
- Git
- GPU recommended for inference/training but CPU will run the web app (slower).
- (Optional) Google Drive access to download the large model files.

---

## Folder layout (relevant parts)

```

lake-plastic-detection/
├── models/                         # model checkpoints (place the large frcnn .pth files here)
│   ├── raw_baseline_yolov8m_768_best.pt
│   ├── aug_yolo11m_1024_best.pt
│   └── names.txt
├── plastic_web2/
│   ├── backend/
│   │   └── app.py                  # FastAPI backend
│   ├── crowd_sourced_images/       # saved uploads
│   └── static/                     # frontend files (index.html, app.js, style.css)
├── requirements.txt
├── README.md
└── .env (see sample below; not checked into git)

````

---

## Installation (local development)

1. **Clone repository**

```bash
git clone https://github.com/<your-username>/lake-plastic-detection.git
cd lake-plastic-detection/lake-plastic-detection
````

2. **Create & activate Python virtual environment**

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

If you plan to use `gdown` to download from Google Drive (recommended), install it:

```bash
pip install gdown
```

---

## Download the large Faster-R-CNN models (from Google Drive)

**Option A — Using browser (recommended for most users)**
Open the Google Drive link above in your browser, locate `frcnn_raw_final.pth` and `frcnn_aug_final.pth`, and download them. Place the downloaded files into the repository `models/` directory:

```
lake-plastic-detection/lake-plastic-detection/models/frcnn_raw_final.pth
lake-plastic-detection/lake-plastic-detection/models/frcnn_aug_final.pth
```

**Option B — Using `gdown` (command-line)**
If you have `gdown` installed you can download the entire folder (if `gdown` supports copying a folder URL) or individual file IDs. Example (folder download):

```bash
# from repo root
cd lake-plastic-detection/lake-plastic-detection
gdown --folder "https://drive.google.com/drive/folders/15SzBP0jWw9KYruj8iMVxtKCvNLVlT3sO?usp=drive_link"
# Move files into models/ if needed
mv frcnn_raw_final.pth models/
mv frcnn_aug_final.pth models/
```

> If `gdown` does not accept the folder direct link in your version, download via browser, or obtain the individual file IDs and run:
>
> ```bash
> gdown --id <file_id> -O models/frcnn_raw_final.pth
> ```

---

## Configure environment variables

Create a `.env` file in the `lake-plastic-detection/lake-plastic-detection/` folder.

**.env (example)**

```env
# Path to the model to use by default (relative or absolute)
MODEL_PATH=models/frcnn_raw_final.pth

# Which backend to prefer: 'frcnn' | 'yolo' | 'auto'
MODEL_BACKEND=frcnn

# Path to class names (one per line)
NAMES_PATH=models/names.txt

# Number of classes (7 for this project)
NUM_CLASSES=7
```

> The backend code loads these using `python-dotenv` (`load_dotenv()`), so you do not need to export them manually if `.env` exists.

---

## Run the web app locally

Start the FastAPI server (from `plastic_web2/backend` folder or point to it):

```bash
cd plastic_web2/backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at: `http://127.0.0.1:8000` — the frontend is served as static files and will call the backend endpoints for predictions.

---

## Quick test & usage

* Use the frontend `index.html` to upload an image and press **Predict**.
* Uploaded images are saved under `plastic_web2/crowd_sourced_images/` and predictions are shown as overlays with counts.
* If EXIF GPS exists in an image, the frontend uses it to prefill location; otherwise the user can select location on the embedded map.

---

## Notes for developers / contributors

### Do **not** commit large `.pth` files to GitHub

* Large model files ( >100 MB) must be hosted off-repo (like the Drive folder above) or tracked with Git LFS.
* Keep `models/` in `.gitignore` if you prefer not to include models in the repo:

Example `.gitignore` additions:

```
# models and uploaded images
models/*.pth
plastic_web2/crowd_sourced_images/
```

### Option: Use Git LFS if you want models versioned in repo

If you prefer to store large models with the repo, use Git LFS. See [git-lfs.github.com](https://git-lfs.github.com/) for installation; then run:

```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes
git commit -m "Track model files with git-lfs"
# then add and push model files (beware GitHub LFS quota)
```

---

## Troubleshooting

* **Model not found error:** Confirm that `MODEL_PATH` in `.env` points to an existing file under `models/`. Use absolute path if needed.
* **Permissions errors saving uploads:** Ensure the `plastic_web2/crowd_sourced_images/` folder exists and is writable.
* **Slow predictions on CPU:** Use a GPU or move to smaller YOLO model (yolov8n) for faster inference on CPU.
* **gdown fails with folder URL:** download manually from Drive via browser.

---

## Testing endpoints (example curl)

If your backend exposes `/predict` (adjust to your actual route):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/image.jpg" \
  -H "accept: application/json"
```

(Adjust `app.py` endpoint names as needed if `/predict` is different.)

---
