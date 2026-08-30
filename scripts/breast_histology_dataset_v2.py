"""
Standalone, importable Dataset for breast_histology specialist v2 training (BreakHis).
Must live in its own module (not a notebook cell) for Windows DataLoader multiprocessing
(num_workers > 0) to work -- worker subprocesses re-import this module fresh, so
ImageFile.LOAD_TRUNCATED_IMAGES must be set here at module level, not just in the notebook.
"""
import random
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BreastHistologyDatasetV2(Dataset):
    """records: list of (path, label_str, subtype, magnification, patient_id)."""

    def __init__(self, records, label_to_idx, transform=None):
        self.records = records
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def _load(self, idx):
        path, label_str, _subtype, _mag, _patient = self.records[idx]
        img = Image.open(path).convert("RGB")
        img.load()
        label = self.label_to_idx[label_str]
        return img, label

    def __getitem__(self, idx):
        try:
            img, label = self._load(idx)
        except (OSError, Image.DecompressionBombError):
            fallback_idx = (idx + 1) % len(self.records)
            img, label = self._load(fallback_idx)
        if self.transform:
            img = self.transform(img)
        return img, label
