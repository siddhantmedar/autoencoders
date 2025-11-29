from dataclasses import dataclass

@dataclass
class cfg:
    d_in = 784
    d_hidden = 256
    d_latent = 64
    num_epochs = 100