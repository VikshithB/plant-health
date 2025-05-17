# train_resnet50.py  ── refined for faster & higher accuracy
import time, json, math, warnings
from pathlib import Path

import torch, torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ── configuration ────────────────────────────────────────────────
DATA_ROOT   = Path("data/PlantDoc-Dataset")
TRAIN_DIR   = DATA_ROOT / "train"
VAL_DIR     = DATA_ROOT / "val"
OUTPUT_DIR  = Path("models")

BATCH_SIZE  = 32
NUM_EPOCHS  = 10            # you asked for 10
BASE_LR     = 3e-4          # starting LR before warm-up
IMG_SIZE    = 224
NUM_WORKERS = 0             # safest on Windows
LABEL_SMOOTHING = 0.1
PATIENCE    = 3             # epochs with no val-loss improv. before LR↓
MIN_LR      = 1e-6
WARMUP_EPOCHS = 1
AMP         = True          # use mixed precision
OUTPUT_DIR.mkdir(exist_ok=True)
BEST_WEIGHTS = OUTPUT_DIR / "resnet50_best.pth"
FINAL_WEIGHTS = OUTPUT_DIR / "resnet50_final.pth"
# ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# data transforms
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
    transforms.ColorJitter(.2, .2, .2, .2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR,   val_tfms)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

num_classes = len(train_ds.classes)
print("Classes:", num_classes)

# model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
scaler = GradScaler(enabled=AMP)

# cosine scheduler with warm-up
def cosine_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    cos = 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) /
                              (NUM_EPOCHS - WARMUP_EPOCHS)))
    return max(cos, MIN_LR / BASE_LR)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

best_acc, epochs_no_improve = 0.0, 0
history = []
start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # ── TRAIN ───────────────────────────────────────
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=AMP):
            out = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_ds)

    # ── VALIDATE ────────────────────────────────────
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(enabled=AMP):
                out = model(xb)
                loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            correct  += (out.argmax(1) == yb).sum().item()
    val_loss /= len(val_ds)
    val_acc  = correct / len(val_ds)

    # scheduler step & early-stopping LR decay
    scheduler.step()
    if history and val_loss >= history[-1]["val_loss"] - 1e-4:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            for pg in optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, MIN_LR)
            epochs_no_improve = 0
            warnings.warn("LR reduced due to plateau")
    else:
        epochs_no_improve = 0

    history.append(dict(epoch=epoch, train_loss=train_loss,
                        val_loss=val_loss, val_acc=val_acc))
    print(f"Epoch {epoch:02}/{NUM_EPOCHS} | "
          f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
          f"val_acc {val_acc*100:.2f}% | lr {optimizer.param_groups[0]['lr']:.2e}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_WEIGHTS)
        print("  🔸 Saved best model")

# save final epoch weights too
torch.save(model.state_dict(), FINAL_WEIGHTS)
with open(OUTPUT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)

mins = (time.time() - start) / 60
print(f"\n✅ Finished in {mins:.1f} min — best val_acc {best_acc*100:.2f}%")
print(f"Best weights → {BEST_WEIGHTS.name}  |  Final weights → {FINAL_WEIGHTS.name}")
