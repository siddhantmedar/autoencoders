import torch
import torch.nn as nn
from config import cfg

class AutoEncoder(nn.Module):
    def __init__(self,d_in):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in,cfg.d_hidden),
            nn.BatchNorm1d(cfg.d_hidden),
            nn.ReLU()
        )

        self.latent_layer = nn.Sequential(
            nn.Linear(cfg.d_hidden,cfg.d_latent),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_latent,cfg.d_hidden),
            nn.BatchNorm1d(cfg.d_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_hidden,d_in),
        )

    def forward(self,x):
        x = x.view(x.size(0),-1)
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)
        reconstructed = self.decoder(latent)

        return reconstructed
    
    