import torch.nn as nn

class SemComModel(nn.Module):
    def __init__(self, vae, pragmatic):
        super().__init__()
        self.vae = vae
        self.pragmatic = pragmatic

    def forward(self, x):
        return self.pragmatic(self.vae(x))
