import torch
import numpy as np
import os

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, cv, patience=5, verbose=False, delta=0, path='checkpoint.pt', save=True):
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

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.save_checkpoint(score, model)
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.save:
            if self.verbose:
                print(f'Validation score increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')

            if self.save_path:
                self.save_path.unlink()
            self.save_path = self.path / ("{}_fold".format(self.cv) + str(np.round(score*100, 2)) + '.pth')
            torch.save(model.state_dict(), self.save_path)

        else:
            pass