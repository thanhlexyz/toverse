import torch.nn as nn

from .conv import Conv
from .bn import BN

class ConvGroup(nn.Module):

    def __init__(self, ci, co):
        super().__init__()
        self.conv1 = Conv(ci, co)
        self.pool  = nn.Conv2d(co, co, kernel_size=2, stride=2, groups=co, bias=False)
        self.bn1   = BN(co)
        self.conv2 = Conv(co, co)
        self.bn2   = BN(co)
        self.act   = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.pool(self.conv1(x))))
        x = self.act(self.bn2(self.conv2(x)))
        return x
