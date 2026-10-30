import torch.nn as nn
import torch

from .conv_group import ConvGroup

class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        BASE = args.base_channel_width

        self.enc1   = ConvGroup(3, BASE)
        self.bottle = nn.Linear(BASE, args.n_latent, bias=False)
        self.act0   = nn.ReLU()

    def forward(self, x):
        x = self.act0(x)
        x = self.enc1(x)
        return self.bottle(x.mean(dim=3).mean(dim=2))
