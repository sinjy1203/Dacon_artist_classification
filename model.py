##
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
## aa

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

##
if __name__ == '__main__':
    model = SimpleNet(5, 100, 500).cuda()
    summary(model, (3, 500, 500))
