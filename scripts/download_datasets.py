import subprocess
import os
from pathlib import Path

DATA_DIR = Path("./data/raw")

def download_kaggle(dataset_id, output_dir):
    """Download from Kaggle and unzip"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📥 Downloading {dataset_id}...")
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset_id,
            "-p", str(output_dir),
            "--unzip"
        ], check=True)
        print(f"✅ {dataset_id} downloaded to {output_dir}")
    except Exception as e:
        print(f"❌ Error downloading {dataset_id}: {e}")

# Week 1 Critical Datasets
datasets = {
    "mariaherrerot/aptos2019": "diabetic_retinopathy",
    "nabeelsajid917/ct-kidney-dataset-normal-cyst-tumor-and-stone": "kidney_stone",
}

print("=" * 60)
print("CliniqAI - Dataset Downloader (Week 1 Critical)")
print("=" * 60)

for kaggle_id, folder_name in datasets.items():
    output_dir = DATA_DIR / folder_name
    download_kaggle(kaggle_id, output_dir)

print("\n" + "=" * 60)
print("✅ Week 1 dataset downloads complete!")
print("=" * 60)