import torch
from dataset import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def validate(model, criterion):
    val_loader = get_dataloaders(split="val")
    total_loss = 0
    total_samples = 0

    for inputs, _ in val_loader:
        batch_size = inputs.size(0)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = inputs.to(device)

        _, _, reconstructed = model(inputs)
        loss = criterion(reconstructed, inputs)
        total_loss += loss.item() * batch_size
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples

    return avg_loss
