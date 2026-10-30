import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class AE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        return self.decoder(self.encoder(x))
