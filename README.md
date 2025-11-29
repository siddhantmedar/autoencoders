# MNIST Autoencoder

A PyTorch implementation of an autoencoder for MNIST digit reconstruction.

## Project Structure

- `model.py` - Autoencoder architecture (encoder ’ latent ’ decoder)
- `train.py` - Training loop with TensorBoard logging and checkpointing
- `evaluate.py` - Validation logic
- `data.py` - MNIST data loading and preprocessing
- `config.py` - Model configuration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model:
```bash
python train.py
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir=runs
```

## Model Architecture

- **Input**: 784 (28x28 flattened MNIST images)
- **Hidden**: 256 units with BatchNorm and ReLU
- **Latent**: 64 dimensions
- **Output**: 784 (reconstructed image)

## Configuration

Edit `config.py` to modify:
- `d_hidden` - Hidden layer size
- `d_latent` - Latent space dimensions
- `num_epochs` - Training epochs
