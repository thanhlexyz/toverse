import torch.nn as nn

from .conv import Conv

class UpBlock(nn.Module):

    def __init__(self, ci, co):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv(ci, co)
        self.bn   = nn.BatchNorm2d(co, eps=1e-12, momentum=0.4)
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(self.up(x))))
