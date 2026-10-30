import torch.nn as nn
import torch

class Discriminator(nn.Module):

    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),)

    def forward(self, z):
        return self.layers(z)
