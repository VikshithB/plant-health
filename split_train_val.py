# split_train_val.py
"""
One-time script:
  • creates data/PlantDoc-Dataset/val/<class>/...
  • moves 15 % of the images from train/<class>/ -> val/<class>/
Run it only ONCE.  If you need to re-do, reclone the dataset.
"""
import random, shutil
from pathlib import Path

DATA_ROOT = Path("data/PlantDoc-Dataset")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
VAL_SPLIT = 0.15
SEED      = 2025

random.seed(SEED)

assert TRAIN_DIR.exists(), f"❌ {TRAIN_DIR} missing"

VAL_DIR.mkdir(exist_ok=True)
moved = 0

for class_dir in TRAIN_DIR.iterdir():
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob("*"))
    n_val  = max(1, int(len(images) * VAL_SPLIT))
    val_images = random.sample(images, n_val)

    (VAL_DIR / class_dir.name).mkdir(parents=True, exist_ok=True)
    for img_path in val_images:
        shutil.move(str(img_path), str(VAL_DIR / class_dir.name / img_path.name))
        moved += 1

print(f"🎉 Moved {moved} images into {VAL_DIR}")
