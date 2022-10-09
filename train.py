##
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from model import *
from dataset import *
from utils import *

##
parser = argparse.ArgumentParser(description="train model")
parser.add_argument('--epochs', default=100, type=int, dest='epochs')
parser.add_argument('--batch_size', default=32, type=int, dest='batch_size')
parser.add_argument('--lr', default=0.001, type=float, dest='lr')
parser.add_argument('--cv', default=5, type=int, dest='cv')
parser.add_argument('--device', default='auto', type=str, dest='device')

parser.add_argument('--root_dir', default='.', type=str, dest='root_dir')

parser.add_argument('--n_blocks', default=2, type=int, dest='n_blocks')
parser.add_argument('--feature', default=100, type=int, dest='feature')
parser.add_argument('--img_shape', default=500, type=int, dest='img_shape')

args = parser.parse_args()
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
CV = args.cv
DEVICE = args.device
N_BLOCKS = args.n_blocks
FEATURE = args.feature
IMG_SHAPE = args.img_shape

##
ROOT_DIR = Path(args.root_dir)

log_dir = ROOT_DIR / 'log'
if log_dir.exists():
    shutil.rmtree(log_dir)
log_dir.mkdir(exist_ok=True)

ckpt_dir = ROOT_DIR / 'ckpt'
if ckpt_dir.exists():
    shutil.rmtree(ckpt_dir)
ckpt_dir.mkdir(exist_ok=True)

##
dataset_train = dataset(data_dir=ROOT_DIR / 'data',
                        transform=transforms.Compose([Scale(IMG_SHAPE), ToTensor()]))

if DEVICE == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(DEVICE)

kfold = KFold(n_splits=CV, shuffle=True)
writer = SummaryWriter(log_dir=str(log_dir))

##
loss_fn = nn.CrossEntropyLoss().to(device)
tonumpy_fn = lambda x: x.detach().cpu().numpy()
pred_fn = lambda x: np.argmax(x, axis=-1)
score_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')

##
final_score = 0
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
    print("Fold {} training".format(fold+1))

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        sampler=train_subsampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        sampler=val_subsampler
    )

    Model = SimpleNet(N_BLOCKS, FEATURE, IMG_SHAPE).to(device)
    optim = torch.optim.Adam(Model.parameters(), lr=LR)
    early_stopping = EarlyStopping(fold, path=ckpt_dir)

    train_iter_num = len(train_loader)
    val_iter_num = len(val_loader)

    for epoch in range(1, EPOCHS+1):
        epoch_train_loss = 0
        epoch_train_score = 0
        epoch_val_loss = 0
        epoch_val_score = 0
        Model.train()

        for train_x, train_y in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            optim.zero_grad()

            output = Model(train_x)
            loss = loss_fn(output, train_y)
            loss.backward()
            optim.step()

            pred = pred_fn(tonumpy_fn(output))
            label = tonumpy_fn(train_y)
            score = score_fn(label, pred)

            epoch_train_loss += loss.item() / train_iter_num
            epoch_train_score += score / train_iter_num

        with torch.no_grad():
            Model.eval()

            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                output = Model(val_x)
                loss = loss_fn(output, val_y)

                pred = pred_fn(tonumpy_fn(output))
                label = tonumpy_fn(val_y)
                score = score_fn(label, pred)

                epoch_val_loss += loss.item() / val_iter_num
                epoch_val_score += score / val_iter_num

        early_stopping(epoch_val_score, Model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.add_scalars("Loss", {str(fold): epoch_train_loss}, epoch)
        writer.add_scalars("Score", {str(fold): epoch_train_score}, epoch)
        writer.add_scalars("Loss", {str(fold): epoch_val_loss}, epoch)
        writer.add_scalars("Score", {str(fold): epoch_val_score}, epoch)

    final_score += early_stopping.best_score / CV

print("Final Macro F1 score: {}".format(final_score))
writer.close()