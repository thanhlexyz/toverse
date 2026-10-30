import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.enc_conv = nn.Conv2d(args.input_shape[1], args.n_channel, 7, padding=3, stride=1)
        self.enc_fc = nn.Linear(args.n_channel * np.prod(args.input_shape[2:]), args.n_latent)

    def forward(self, x):
        x = F.relu(self.enc_conv(x))
        x_flat = x.flatten(1)
        fc_out = self.enc_fc(x_flat)
        return fc_out

class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dec_fc = nn.Linear(args.n_latent, args.n_channel * np.prod(args.input_shape[2:]))
        self.dec_deconv = nn.ConvTranspose2d(args.n_channel, args.input_shape[1], 3, stride=1, padding=1)

    def forward(self, z):
        args = self.args
        z_flat = z.flatten(start_dim=1)
        x = F.relu(self.dec_fc(z_flat))
        x = x.view(-1, args.n_channel, *args.input_shape[2:])
        x = F.relu(self.dec_deconv(x))
        return x

class VAE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


if __name__ == '__main__':
    model = VAE()
    x = torch.randn(1, 1, 28, 28)
    print(f'{x.shape=}')
    x_hat = model(x)
