"""
Dataset class for Router Stage 2 (modality classifier).

Lives in its own file so multiprocessing DataLoader workers can import it --
required on Windows for num_workers>0 to work at all.
"""

import random
from torch.utils.data import Dataset
from PIL import Image, ImageFile

# Each DataLoader worker is a separate process that imports this module fresh --
# setting this here (module level) ensures every worker gets it, not just the
# main notebook process.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModalityDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.paths = dataframe["filepath"].tolist()
        self.labels = dataframe["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Pre-validation in the notebook should catch nearly everything, but this
        # is the actual safety net: if a file still fails to load at train time
        # (for any reason), swap in a different random sample instead of crashing
        # the whole epoch. Training should never die on one bad file.
        for _ in range(5):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, self.labels[idx]
            except Exception as e:
                print(f"[warning] failed to load {self.paths[idx]}: {e} -- substituting a different sample")
                idx = random.randint(0, len(self.paths) - 1)

        raise RuntimeError(
            "Repeated failures loading images from this dataset -- "
            "there may be widespread corruption beyond isolated bad files."
        )