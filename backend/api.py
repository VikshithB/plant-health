# backend/api.py  – fully updated (handles slightly-corrupt JPEGs & nicer errors)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True       # ← lets Pillow open imperfect JPEGs

import torch, torchvision
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F

# ── paths & constants ────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
WEIGHTS    = ROOT / "models" / "resnet50_best.pth"
TRAIN_DIR  = ROOT / "data" / "PlantDoc-Dataset" / "train"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class names in canonical order (28 training classes)
CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])

# ── model ────────────────────────────────────────────────────────
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.to(DEVICE).eval()

# image transforms (same as training/val)
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Detector API",
    version="1.0",
    description="ResNet-50 model trained on 28 PlantDoc classes",
)

# allow local Streamlit (http://localhost:8501) – tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────
def read_image(file_bytes: bytes) -> Image.Image:
    """Decode image bytes → PIL.Image or raise 400."""
    try:
        img = Image.open(BytesIO(file_bytes))
        img = img.convert("RGB")          # ensure 3-channel
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def predict(img: Image.Image, topk: int = 3):
    """Return top-k predictions [{class, prob}, …]."""
    xb = tfms(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(xb)
        probs = F.softmax(logits, 1)[0]
    top = probs.topk(topk)
    return [
        {"class": CLASS_NAMES[int(i)], "prob": round(float(p), 4)}
        for i, p in zip(top.indices, top.values)
    ]


# ── route ────────────────────────────────────────────────────────
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    img = read_image(await file.read())
    results = predict(img, topk=3)
    return JSONResponse(content={"predictions": results})
