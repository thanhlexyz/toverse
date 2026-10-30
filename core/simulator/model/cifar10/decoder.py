import torch.nn as nn

from .up_block import UpBlock
from .conv import Conv

class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        BASE = args.base_channel_width
        self.dec_fc = nn.Linear(args.n_latent, BASE * 16 * 16, bias=True)
        self.dec1   = UpBlock(BASE, BASE)
        self.recon  = Conv(BASE, 3)

    def forward(self, z):
        BASE = self.args.base_channel_width
        x = self.dec_fc(z).reshape(z.shape[0], BASE, 16, 16)
        x = self.dec1(x)
        return self.recon(x)
