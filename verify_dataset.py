# verify_dataset.py (patched)
import os, random, glob
from pathlib import Path

DATA_ROOT = Path("data/PlantDoc-Dataset")
SEED = 42
SAMPLE_PER_SPLIT = 3

random.seed(SEED)

def inspect_split(split_dir: Path):
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    img_counts, total = {}, 0
    for cdir in class_dirs:
        n = len(list(cdir.glob("*")))
        img_counts[cdir.name] = n
        total += n
    all_imgs = [p for c in class_dirs for p in c.glob("*")]
    samples = random.sample(all_imgs, min(SAMPLE_PER_SPLIT, len(all_imgs)))
    return class_dirs, img_counts, total, samples

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"❌ Could not find {DATA_ROOT}")

    for split in ["train", "test"]:
        split_path = DATA_ROOT / split
        if not split_path.exists():
            print(f"⚠️  Split '{split}' missing — skipping.")
            continue

        print(f"\n===== {split.upper()} SPLIT =====")
        class_dirs, img_counts, total_imgs, samples = inspect_split(split_path)
        print(f"✅ {len(class_dirs)} classes")
        print(f"✅ {total_imgs} total images")

        largest = sorted(img_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 largest classes (name, count):")
        for name, cnt in largest:
            print(f"  • {name:<30} {cnt:>5}")

        print("\nRandom sample image paths:")
        for p in samples:
            try:
                rel = p.relative_to(Path.cwd())
            except ValueError:
                rel = p
            print(f"  • {rel}")

    print("\n🎉 Dataset structure looks good!")

if __name__ == "__main__":
    main()
