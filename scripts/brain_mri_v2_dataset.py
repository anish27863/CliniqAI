"""
brain_mri_v2_dataset.py

Standalone, importable Dataset module for the CliniqAI brain_mri specialist
retrain. Lives under scripts/ per project convention — Windows DataLoader
multiprocessing silently falls back to single-process if the Dataset class
is defined inline in a notebook cell instead of a real .py file.

Usage from the notebook:
    from scripts.brain_mri_v2_dataset import BrainMRIDataset, build_samples, validate_images
"""

import os
import random
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Module-level, not just in the notebook — required so truncated JPEGs
# don't crash the worker processes.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_samples(root_dir):
    """
    Walk an ImageFolder-style directory (one subfolder per class) and return
    (samples, class_names) where samples is a list of (path, label_idx) and
    class_names is the sorted list of subfolder names — derived directly from
    disk, never manually re-typed, to avoid the label-order bugs already hit
    twice elsewhere in this project (kidney_ct, pneumonia).
    """
    root_dir = Path(root_dir)
    class_names = sorted(
        [d.name for d in root_dir.iterdir() if d.is_dir()]
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for class_name in class_names:
        class_dir = root_dir / class_name
        for f in class_dir.rglob("*"):
            if f.suffix.lower() in IMG_EXTENSIONS:
                samples.append((str(f), class_to_idx[class_name]))

    return samples, class_names, class_to_idx


def validate_images(samples, verbose=True):
    """
    Pre-validation pass: try to actually load every image once and drop any
    that fail. Cheaper to do this once up front than to hit it mid-epoch.
    Returns a filtered samples list.
    """
    good = []
    bad = []
    for path, label in samples:
        try:
            with Image.open(path) as img:
                img.load()
            good.append((path, label))
        except Exception as e:
            bad.append((path, str(e)))

    if verbose:
        print(f"[validate_images] {len(good)} OK, {len(bad)} corrupted/unreadable")
        for path, err in bad[:20]:
            print(f"  BAD: {path} -> {err}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")

    return good


class BrainMRIDataset(Dataset):
    """
    ImageFolder-style dataset with a runtime fallback: if a sample somehow
    fails to load at __getitem__ time despite pre-validation (e.g. disk
    hiccup), substitute a different random sample instead of crashing the
    whole training run.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback: substitute a different random sample rather than crash.
            alt_idx = random.randrange(len(self.samples))
            alt_path, label = self.samples[alt_idx]
            img = Image.open(alt_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label
