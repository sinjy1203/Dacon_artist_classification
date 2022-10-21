##
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import *

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
from torchvision import transforms
from torchvision import datasets

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # dataset_ = dataset(Path('data'), label=False, transform=transforms.Compose([Scale(500), ToTensor()]))
    # dataset_ = dataset(
    #     Path('data'),
    #     transform=transforms.Compose([
    #         transforms.RandomResizedCrop(size=(300, 300)),
    #         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    #         transforms.RandomRotation(45),
    #         transforms.ToTensor()
    #     ]))
    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    #
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # dataset_ = datasets.CIFAR100('.', download=True, train=True, transform=transform_train)
    #
    # img, label = dataset_.__getitem__(1)
    # img_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    # img_aug = transforms.RandomHorizontalFlip(0.5)
    # img_aug = transforms.RandomPerspective(distortion_scale=0.3, p=1)
    # img_aug = transforms.RandomResizedCrop(size=(300, 300))
    # img_aug = transforms.RandomRotation(45)
    # img_ = img_aug(img)
    # print(np.array(img_).shape)
    # plt.subplot(121)
    # print(np.array(img).shape)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(img_)
    # plt.show()
    # print(img.shape)
    dataset_train = dataset(
        data_dir=Path('.') / 'data',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(size=(500, 500)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomRotation(45),
            ToTensor()
        ]))
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=8
    )

    x, y = next(iter(train_loader))
    x_ = x.clone().detach().numpy().transpose((0,2,3,1))
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(x.size()[0])
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    x = x.numpy().transpose((0,2,3,1))
    # kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    #
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
    #     print(val_ids)
    # Path('./ckpt2').mkdir(exist_ok=True)