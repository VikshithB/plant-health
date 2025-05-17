# evaluate_test.py  – works when train has 28 classes & test has 27
"""
Reports accuracy on data/PlantDoc-Dataset/test and saves a 28×28 confusion matrix.

Outputs:
  • prints overall accuracy + per-class report
  • models/confusion_matrix.png (28×28)
"""

from pathlib import Path
import numpy as np
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────
TRAIN_DIR = Path("data/PlantDoc-Dataset/train")
TEST_DIR  = Path("data/PlantDoc-Dataset/test")
WEIGHTS   = Path("models/resnet50_best.pth")
FIG_PATH  = Path("models/confusion_matrix.png")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 224
BATCH_SZ  = 32
# ───────────────────────────────────────────────────────

# 1⃣  canonical 28-class order (from train)
train_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
train_cls2idx = {c: i for i, c in enumerate(train_classes)}

# 2⃣  test dataset & mapping 27->28 indices
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
test_ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
test_loader = DataLoader(test_ds, batch_size=BATCH_SZ,
                         shuffle=False, num_workers=0, pin_memory=True)

test_to_train_idx = [train_cls2idx[c] for c in test_ds.classes]

# 3⃣  build model with 28 outputs & load weights
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(train_classes))   # 28
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.to(DEVICE).eval()

# 4⃣  inference
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb).argmax(1).cpu().numpy()
        all_preds.extend(out)
        # map 0-26 test labels → 0-27 train labels
        mapped = [test_to_train_idx[i] for i in yb.numpy()]
        all_true.extend(mapped)

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
acc = (all_preds == all_true).mean()
print(f"\n✅  Test accuracy: {acc*100:.2f}% "
      f"({(all_preds==all_true).sum()}/{len(all_true)})")

print("\nPer-class precision / recall / F1:")
# replace the classification_report call in evaluate_test.py
print(classification_report(
        all_true,
        all_preds,
        labels=list(range(len(train_classes))),     # ← add this line
        target_names=train_classes,
        digits=2,
        zero_division=0))                           # ← avoids divide‑by‑zero warnings


# 5⃣  confusion matrix (28×28)
cm = confusion_matrix(all_true, all_preds, labels=range(len(train_classes)))
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm)
ax.set_title("Confusion matrix – PlantDoc test set")
ax.set_xticks(range(len(train_classes)))
ax.set_yticks(range(len(train_classes)))
ax.set_xticklabels(range(len(train_classes)), rotation=90, fontsize=6)
ax.set_yticklabels(range(len(train_classes)), fontsize=6)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
print(f"📊 Confusion matrix saved → {FIG_PATH}")
