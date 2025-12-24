import tomllib
from pathlib import Path
import torch
import torch.nn as nn

CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)


class AutoEncoder(nn.Module):
    def __init__(self, d_in):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in, cfg["model"]["d_hidden"]),
            nn.BatchNorm1d(cfg["model"]["d_hidden"]),
            nn.ReLU(),
        )

        self.latent_layer = nn.Sequential(
            nn.Linear(cfg["model"]["d_hidden"], cfg["model"]["d_latent"]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg["model"]["d_latent"], cfg["model"]["d_hidden"]),
            nn.BatchNorm1d(cfg["model"]["d_hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["model"]["d_hidden"], d_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)
        reconstructed = self.decoder(latent)

        return encoded, latent, reconstructed
