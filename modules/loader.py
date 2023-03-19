import os

import torch
import numpy as np
from torch.utils.data import dataloader, Dataset
import pandas as pd

PATH = "data/dhs_train/IA-2015-7-00010016.npz"
PATH_2 = "../data/dhs_train"
CSV = "../data/train.csv"


class CustomDataset(Dataset):
    def __init__(self, path: str, target: str, transform=None):
        self.path = path
        self.files = os.listdir(path)
        self.targets = pd.read_csv(target)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.path, self.files[idx])
        data = np.load(path)
        image = data['x']
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        target = self.targets.iloc[idx]["water_index"]
        return image, target


# if __name__ == "__main__":
#     l = CustomDataset(PATH_2, CSV)
#     df = pd.read_csv(CSV)
#     print(l[2])
