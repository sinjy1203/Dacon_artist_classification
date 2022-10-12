##
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
import glob

import torch
import torch.nn as nn
from torchvision import transforms

from model import *
from dataset import *
from utils import *

##
parser = argparse.ArgumentParser(description="test model")
parser.add_argument('--epochs', default=100, type=int, dest='epochs')
parser.add_argument('--batch_size', default=32, type=int, dest='batch_size')
parser.add_argument('--lr', default=0.001, type=float, dest='lr')
parser.add_argument('--cv', default=5, type=int, dest='cv')
parser.add_argument('--device', default='auto', type=str, dest='device')

parser.add_argument('--root_dir', default='.', type=str, dest='root_dir')

parser.add_argument('--n_blocks', default=2, type=int, dest='n_blocks')
parser.add_argument('--feature', default=100, type=int, dest='feature')
parser.add_argument('--img_shape', default=500, type=int, dest='img_shape')

##
args = parser.parse_args()
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
CV = args.cv
DEVICE = args.device
N_BLOCKS = args.n_blocks
FEATURE = args.feature
IMG_SHAPE = args.img_shape
ROOT_DIR = Path(args.root_dir)
# N_BLOCKS = 2
# FEATURE = 100
# IMG_SHAPE = 100
# ROOT_DIR = Path('.')
# ROOT_DIR_ = Path('C:/Users/sinjy/Desktop')
# BATCH_SIZE = 32

##
dataset_test = dataset(data_dir=ROOT_DIR / 'data', label=False,
                        transform=transforms.Compose([Scale(IMG_SHAPE), ToTensor()]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.CrossEntropyLoss().to(device)
tonumpy_fn = lambda x: x.detach().cpu().numpy()
pred_fn = lambda x: np.argmax(x, axis=-1)

## ensemble model to predict
pred_total = []
with torch.no_grad():
    for fold, model_path in enumerate(glob.glob(str(ROOT_DIR / 'ckpt' / '*')), start=1):
        loader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE)
        # Model = SimpleNet(N_BLOCKS, FEATURE, IMG_SHAPE).to(device)
        Model = ResNet(freeze=True).to(device)
        Model.load_state_dict(torch.load(model_path))

        Model.eval()
        pred = []
        for test_x in tqdm(loader_test, desc='Fold {}'.format(fold)):
            test_x = test_x.to(device)

            output = Model(test_x)
            pred_prob = tonumpy_fn(output)
            pred += [pred_prob]
        pred_total += [np.concatenate(pred, axis=0)]

##
pred_indices = np.mean(np.stack(pred_total, axis=-1), axis=-1).argmax(axis=-1)
pred_object = dataset_test.le.inverse_transform(pred_indices)
sub = pd.read_csv(ROOT_DIR / 'data' / 'sample_submission.csv')
sub['artist'] = pred_object

##
pred_dir = Path(ROOT_DIR / 'pred')
pred_dir.mkdir(exist_ok=True)
sub.to_csv(pred_dir / 'submission.csv', index=False)
