"""
Dataset class for Router Stage 1 (medical vs non-medical).

This lives in its own file, not inside the notebook, because Windows'
multiprocessing spawns worker processes that need to `import` your Dataset
class from a real module -- a class defined inline in a notebook cell can't
be pickled/imported by worker processes.
"""

import random
from torch.utils.data import Dataset
from PIL import Image, ImageFile

# Each DataLoader worker is a separate process that imports this module fresh --
# setting this here (module level) ensures every worker gets it.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GatekeeperDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.paths = dataframe["filepath"].tolist()
        self.labels = dataframe["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Pre-validation in the notebook should catch nearly everything, but this
        # is the actual safety net against any file that still fails at train time.
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