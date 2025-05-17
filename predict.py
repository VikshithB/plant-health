# predict.py
"""
Usage:
    python predict.py <path_to_leaf_image>

Outputs:
    top-3 predicted classes + probabilities
"""
import sys, torch, torchvision
from pathlib import Path
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F

IMG_SIZE = 224
WEIGHTS  = Path("models/resnet50_best.pth")
CLASSES  = sorted((Path("data/PlantDoc-Dataset/train")).glob("*"))
CLASS_NAMES = [p.name for p in CLASSES]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.to(device).eval()

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

def predict(img_path: Path, k=3):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    xb = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(xb)
        probs = F.softmax(out, dim=1)[0].cpu()
    topk = probs.topk(k)
    for idx, prob in zip(topk.indices, topk.values):
        print(f"{CLASS_NAMES[idx]:<35}  {prob*100:5.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image>")
        sys.exit(1)
    predict(Path(sys.argv[1]))
