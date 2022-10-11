##
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models

##
class SimpleNet(nn.Module):
    def __init__(self, n_hidden_block, n_feature, img_size):
        super(SimpleNet, self).__init__()
        self.fc_feature = n_feature * int(img_size / 2**(n_hidden_block+1)) ** 2

        def CBADP(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                  bias=True, p=0.5):
            layers = []
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)]
            layers += [nn.BatchNorm2d(out_ch)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(p)]
            layers += [nn.MaxPool2d(2)]

            layer = nn.Sequential(*layers)
            return layer

        block_lst = [CBADP(3, n_feature)]
        block_lst += [CBADP(n_feature, n_feature) for _ in range(n_hidden_block)]
        self.block_lst = nn.Sequential(*block_lst)

        fc_block = []
        fc_block += [nn.Linear(self.fc_feature, n_feature)]
        fc_block += [nn.ReLU()]
        fc_block += [nn.Dropout(0.5)]
        fc_block += [nn.Linear(n_feature, 50)]
        self.fc_block = nn.Sequential(*fc_block)

    def forward(self, x):
        x = self.block_lst(x)
        x = x.view(-1, self.fc_feature)
        x = self.fc_block(x)
        return x

class ResNet(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet, self).__init__()
        if freeze:
            self.net = models.resnet50(pretrained=True)
            for param in self.net.parameters():
                param.requires_grad = False
        else:
            self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 50)

    def forward(self, x):
        return self.net(x)



##
from sklearn.metrics import f1_score
# loss_fn = nn.CrossEntropyLoss().to(device)
tonumpy_fn = lambda x: x.detach().cpu().numpy()
pred_fn = lambda x: np.argmax(x, axis=-1)
score_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
if __name__ == '__main__':
    model = SimpleNet(5, 100, 500).cuda()
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = nn.Linear(model.fc.in_features, 50)
    # model = ResNet(freeze=True)
    model = model.to(device='cuda')
    summary(model, (3, 500, 500))
    # Model = SimpleNet(5, 100, 500)
    # output = Model(torch.rand((10, 3, 500, 500)))
    # pred = pred_fn(tonumpy_fn(output))
    # score = score_fn(np.random.randint(50, size=10), pred)
    # print(score)
