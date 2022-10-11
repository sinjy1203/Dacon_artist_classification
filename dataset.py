##
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

##
class dataset(Dataset):
    def __init__(self, data_dir, label=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.data_csv = pd.read_csv(data_dir / 'train.csv')
        self.le = LabelEncoder()
        self.le.fit(self.data_csv['artist'])
        if not label:
            self.data_csv = pd.read_csv(data_dir / 'test.csv')
        self.label = label

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        data_row = self.data_csv.iloc[idx]
        img = PIL.Image.open(self.data_dir / data_row['img_path'])
        if self.label:
            label = self.le.transform([data_row['artist']])[0]
            label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        if self.label:
            return img, label
        else:
            return img

class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img):
        img = self.totensor(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

class Scale(object):
    def __init__(self, shape):
        self.scale = transforms.Resize((shape, shape))

    def __call__(self, img):
        return self.scale(img)


##
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    dataset_ = dataset(Path('data'), label=False, transform=transforms.Compose([Scale(500), ToTensor()]))
    # dataset_ = dataset(Path('data'))

    img = dataset_.__getitem__(0)
    # print(img.shape)
    # plt.imshow(data['img'])
    # plt.show()
    print(img.shape)
