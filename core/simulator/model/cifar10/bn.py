import torch.nn as nn

class BN(nn.BatchNorm2d):
    def __init__(self, c):
        super().__init__(c, eps=1e-12, momentum=0.4)
        self.weight.data.fill_(1.)
        self.weight.requires_grad = False
