import torch.nn as nn
import torch

from .conv_group import ConvGroup

class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        BASE = args.base_channel_width

        self.cls1     = ConvGroup(3,      BASE)
        self.cls_head = nn.Linear(BASE, 10, bias=False)
        self.cls_temp = 1. / 9

    def forward(self, recon):
        x = self.cls1(recon)
        return self.cls_head(x.mean(dim=3).mean(dim=2)) * self.cls_temp
