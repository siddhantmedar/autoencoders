import tomllib
from pathlib import Path
import torch.nn as nn

CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)


class AutoEncoder(nn.Module):
    def __init__(self, d_in):
        super().__init__()

        d_hidden1 = cfg["model"]["d_hidden1"]
        d_hidden2 = cfg["model"]["d_hidden2"]
        d_latent = cfg["model"]["d_latent"]

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden1),
            nn.BatchNorm1d(d_hidden1),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_hidden2),
            nn.BatchNorm1d(d_hidden2),
            nn.ReLU(),
        )

        self.latent_layer = nn.Sequential(
            nn.Linear(d_hidden2, d_latent),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden2),
            nn.BatchNorm1d(d_hidden2),
            nn.ReLU(),
            nn.Linear(d_hidden2, d_hidden1),
            nn.BatchNorm1d(d_hidden1),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_in),
            nn.Sigmoid(),
        )

    def dec_forward(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)
        reconstructed = self.decoder(latent)

        return encoded, latent, reconstructed
