import torch
import numpy as np
import os

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(x, y, model, criterion, device):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(x.size()[0]).to(device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    # compute output
    output = model(x)
    loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    return loss

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, cv, path, patience=10, verbose=False, delta=0, save=True):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save_path = None
        self.save = save
        self.cv = cv

    def __call__(self, val_score, model, optim, epoch, lr_scheduler):
        score = val_score

        if self.best_score is None:
            self.save_checkpoint(score, model, optim, epoch, lr_scheduler)
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, optim, epoch, lr_scheduler)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, optim, epoch, lr_scheduler):
        if self.save:
            if self.verbose:
                print(f'Validation score increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')

            if self.save_path:
                self.save_path.unlink()
            self.save_path = self.path / ("{}fold_{}_resume".format(self.cv, str(np.round(score, 2))) + '.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'best_score': self.best_score,
                'scheduler_state_dict': lr_scheduler.state_dict()
            }, self.save_path)

        else:
            pass
