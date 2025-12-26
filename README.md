# MNIST Autoencoder

A PyTorch implementation of an autoencoder for MNIST digit reconstruction with 2D latent space visualization.

## Project Structure

- `model.py` - Autoencoder architecture (encoder → latent → decoder)
- `run.py` - Training script with CLI arguments and checkpointing
- `dataset.py` - MNIST data loading using HuggingFace datasets
- `config.toml` - Model and training configuration
- `notebooks/` - Analysis and visualization notebooks
- `trained_models/` - Pre-trained model weights

## Installation

```bash
uv sync
```

## Usage

Train the model:
```bash
python run.py --train
python run.py --train --epochs 50 --lr 1e-3 --batch_size 128
python run.py --train --save_dir trained_models
```

Test the model:
```bash
python run.py  # loads checkpoints/best.pt by default
```

Load pre-trained model:
```python
from model import AutoEncoder
import torch

model = AutoEncoder(d_in=784)
checkpoint = torch.load("trained_models/model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Model Architecture

- **Input**: 784 (28x28 flattened MNIST images)
- **Encoder**: 784 → 256 → 32 (with BatchNorm + ReLU)
- **Latent**: 2 dimensions (for visualization)
- **Decoder**: 2 → 32 → 256 → 784 (with Sigmoid output)

## Configuration

Edit `config.toml` to modify:
- `d_hidden1` - First hidden layer size (default: 256)
- `d_hidden2` - Second hidden layer size (default: 32)
- `d_latent` - Latent space dimensions (default: 2)
- `num_epochs` - Training epochs (default: 100)
- `batch_size` - Batch size (default: 64)

## Latent Space Analysis

The notebook `notebooks/analyze_latent_space.ipynb` contains a detailed study of how the autoencoder learns to organize digits in 2D latent space.

### Training with Latent Tracking

We train the autoencoder with a **2-dimensional latent space** (instead of the typical higher dimensions) specifically to enable visualization. During training, we record the latent representations of all 60,000 training samples at each epoch, storing both the 2D coordinates and their corresponding digit labels.

### Latent Space Evolution

![Latent Space Evolution](docs/latent_evolution.gif)

**Key observations:**
- **Early training**: Points are randomly scattered with no clear structure
- **Mid-training**: Digit classes begin forming distinct clusters
- **Late training**: Clear separation between most digit classes, though some overlap remains (e.g., 4/9, 3/5/8)

### Sampling from Latent Space

By sampling random points from the latent space and decoding them, we can explore what the model has learned:

**Separated regions**: Sampling from well-separated clusters (e.g., where "1"s cluster) produces clear, recognizable digits.

**Overlapping regions**: Sampling from areas where multiple digit classes overlap produces ambiguous or hybrid images - blends of multiple digits that reveal the model's uncertainty in those regions.

This demonstrates that the autoencoder learns a meaningful continuous representation where similar digits are placed nearby, and interpolating between clusters produces smooth transitions between digit styles.
