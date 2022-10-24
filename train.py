##
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import gc
import os
import stat

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
import warnings
warnings.filterwarnings('ignore')

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
parser.add_argument('--cutmix_prob', default=0.5, type=float, dest='cutmix_prob')
parser.add_argument('--train_fold', nargs='+', type=int, dest='train_fold')

parser.add_argument('--del_ckpt', default=False, type=bool, dest='del_ckpt')

args = parser.parse_args()
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
CV = args.cv
DEVICE = args.device
N_BLOCKS = args.n_blocks
FEATURE = args.feature
IMG_SHAPE = args.img_shape
CUTMIX_PROB = args.cutmix_prob

##
ROOT_DIR = Path(args.root_dir)


def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

log_dir = ROOT_DIR / 'log'
if log_dir.exists():
    shutil.rmtree(log_dir, onerror=remove_readonly)
log_dir.mkdir(exist_ok=True)

ckpt_dir = ROOT_DIR / 'ckpt'
if args.del_ckpt and ckpt_dir.exists():
    shutil.rmtree(ckpt_dir, onerror=remove_readonly)
ckpt_dir.mkdir(exist_ok=True)

##
dataset_train = dataset(
    data_dir=ROOT_DIR / 'data',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(size=(IMG_SHAPE, IMG_SHAPE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(45),
        ToTensor()
    ]))

dataset_valid = dataset(data_dir=ROOT_DIR / 'data',
                        transform=transforms.Compose([Scale(IMG_SHAPE), ToTensor()]))

if DEVICE == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(DEVICE)

kfold = KFold(n_splits=CV, shuffle=True, random_state=1)
writer = SummaryWriter(log_dir=str(log_dir))

##
loss_fn = nn.CrossEntropyLoss().to(device)
tonumpy_fn = lambda x: x.detach().cpu().numpy()
pred_fn = lambda x: np.argmax(x, axis=-1)
score_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')

##
final_score = 0
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
    if fold not in args.train_fold:
        continue
    print("Fold {} training".format(fold))

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        sampler=train_subsampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=BATCH_SIZE,
        sampler=val_subsampler
    )

    # Model = SimpleNet(N_BLOCKS, FEATURE, IMG_SHAPE).to(device)
    # Model = ResNet(freeze=False).to(device)
    Model = EfficientNet_v2().to(device)
    optim = torch.optim.Adam(Model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.85)
    start_epoch = 1

    early_stopping = EarlyStopping(fold, path=ckpt_dir)

    if glob.glob(str(ckpt_dir / '{}fold_*_resume.pth'.format(fold))):
        model_path = glob.glob(str(ckpt_dir / '{}fold_*_resume.pth'.format(fold)))[0]
        model_data = torch.load(model_path)
        Model.load_state_dict(model_data['model_state_dict'])
        optim.load_state_dict(model_data['optim_state_dict'])
        lr_scheduler.load_state_dict(model_data['scheduler_state_dict'])
        start_epoch = model_data['epoch'] + 1
        early_stopping.save_path = Path(model_path)

    train_iter_num = len(train_loader)
    val_iter_num = len(val_loader)

    for epoch in range(start_epoch, EPOCHS+1):
        epoch_train_loss = 0
        epoch_train_score = 0
        epoch_val_loss = 0
        epoch_val_score = 0
        Model.train()

        for train_x, train_y in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            optim.zero_grad()

            r = np.random.rand(1)
            if r < CUTMIX_PROB:
                loss = cutmix(train_x, train_y, Model, loss_fn, device)
            else:
                output = Model(train_x)
                loss = loss_fn(output, train_y)

            loss.backward()
            optim.step()

            epoch_train_loss += loss.item() / train_iter_num

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
        
        lr_scheduler.step()
        early_stopping(epoch_val_score, Model, optim, epoch, lr_scheduler)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.add_scalars("fold {} Loss".format(fold), {"train": epoch_train_loss}, epoch)
        writer.add_scalars("fold {} Loss".format(fold), {"valid": epoch_val_loss}, epoch)
        writer.add_scalars("fold {} Score".format(fold), {"valid": epoch_val_score}, epoch)

        del train_x, train_y, val_x, val_y
        gc.collect()

    os.rename(str(early_stopping.save_path), str(early_stopping.save_path).replace("_resume", ""))

    final_score += early_stopping.best_score / len(args.train_fold)
    del Model
    gc.collect()

print("Final Macro F1 score: {}".format(final_score))
writer.close()