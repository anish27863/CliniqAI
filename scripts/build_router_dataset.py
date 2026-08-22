"""
Build the router training dataset by sampling images from each
specialist dataset. The router only needs to distinguish modalities,
not diagnose, so a few hundred images per class is enough.
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

RAW_DIR = Path("./data/raw")
ROUTER_DIR = Path("./data/router_training")
SAMPLES_PER_CLASS = 300

# Map: router class name -> source folder
SOURCE_MAP = {
    "diabetic_retinopathy": RAW_DIR / "diabetic_retinopathy" / "train_images" / "train_images",
    "kidney_ct": RAW_DIR / "kidney_stone" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    "non_medical": RAW_DIR / "cifar10" / "cifar-10-batches-py",
}

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_images(folder: Path, limit: int) -> list:
    """Recursively find image files in a folder, return a random sample."""
    all_images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTENSIONS:
                all_images.append(Path(root) / f)
    random.shuffle(all_images)
    return all_images[:limit]


def build_dataset():
    print("=" * 60)
    print("Building Router Training Dataset (CliniqAI)")
    print("=" * 60)

    for class_name, source_folder in SOURCE_MAP.items():
        if not source_folder.exists():
            print(f"⚠️  Skipping {class_name}: {source_folder} not found")
            continue

        dest_folder = ROUTER_DIR / class_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        images = find_images(source_folder, SAMPLES_PER_CLASS)
        print(f"\n📂 {class_name}: found {len(images)} images (target {SAMPLES_PER_CLASS})")

        for i, img_path in enumerate(images):
            dest_path = dest_folder / f"{class_name}_{i:04d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dest_path)

        print(f"✅ Copied {len(images)} images to {dest_folder}")

    print("\n" + "=" * 60)
    print("✅ Router dataset build complete!")
    print("=" * 60)
    print("\nClass distribution:")
    total = 0
    for class_name in SOURCE_MAP:
        class_dir = ROUTER_DIR / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*")))
            print(f"  {class_name}: {count} images")
            total += count
    print(f"\nTotal: {total} images")


if __name__ == "__main__":
    build_dataset()