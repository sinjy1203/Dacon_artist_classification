##
import numpy as np
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import *
from dataset import *

##
parser = argparse.ArgumentParser(description="train model")
parser.add_argument('--epochs', default=100, type=int, dest='epochs')
parser.add_argument('--batch_size', default=32, type=int, dest='batch_size')

parser.add_argument('--n_blocks', default=2, type=int, dest='n_blocks')
parser.add_argument('--feature', default=100, type=int, dest='feature')

args = parser.parse_args()
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
N_BLOCKS = args.n_blocks
FEATURE = args.feature

##
log_dir = Path.cwd() / 'log'
log_dir.mkdir(exist_ok=True)
ckpt_dir = Path.cwd() / 'ckpt'
ckpt_dir.mkdir(exist_ok=True)

##
dataset_train = dataset(data_dir=Path.cwd() / 'data',
                        transform=transforms.Compose([ToTensor(), Scale(500)]))
loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE)

Model = SimpleNet(N_BLOCKS, FEATURE, 500)
