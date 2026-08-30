"""
Standalone, importable Dataset for Router Stage 1 v2/v3 training.
Must live in its own module (not a notebook cell) for Windows DataLoader
multiprocessing (num_workers > 0) to work -- worker subprocesses re-import
this module fresh, so ImageFile.LOAD_TRUNCATED_IMAGES must be set here at
module level, not just in the training notebook.
"""
import random
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GatekeeperDatasetV2(Dataset):
    """records: list of (path, label_str, source_tag). label_str in {'medical','negative'}."""

    def __init__(self, records, label_to_idx, transform=None):
        self.records = records
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def _load(self, idx):
        path, label_str, _tag = self.records[idx]
        img = Image.open(path).convert("RGB")
        img.load()
        label = self.label_to_idx[label_str]
        return img, label

    def __getitem__(self, idx):
        try:
            img, label = self._load(idx)
        except (OSError, Image.DecompressionBombError):
            # Runtime fallback: corrupted/unreadable file -- substitute a random
            # different sample instead of crashing the whole training run.
            fallback_idx = (idx + 1) % len(self.records)
            img, label = self._load(fallback_idx)
        if self.transform:
            img = self.transform(img)
        return img, label


class RandomResolutionDegradation:
    """Randomly downsamples then upscales back, breaking the 'blocky texture -> medical'
    shortcut the v2 gatekeeper learned from breast_histology's 50px patches upscaled 4.5x.
    Applied only in the train transform -- eval/test/generalization stay untouched so existing
    per-source accuracy numbers remain comparable."""

    def __init__(self, p=0.5, low_res_range=(32, 96)):
        self.p = p
        self.low_res_range = low_res_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        low = random.randint(*self.low_res_range)
        img = img.resize((low, low), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img
