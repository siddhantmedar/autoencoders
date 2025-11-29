import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def fetch_dataset(split=None, val_ratio=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # mean and std for mnist
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    # Split test set into validation and test sets
    val_size = int(len(test_dataset) * val_ratio)
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if split=="train":
        return train_loader
    elif split=="val":
        return val_loader
    elif split=="test":
        return test_loader
    

if __name__ == "__main__":
    train_loader = fetch_dataset("train")
    val_loader = fetch_dataset("val")
    test_loader = fetch_dataset("test")

    print(f'train_loader batches: {len(train_loader)}')
    print(f'val_loader batches: {len(val_loader)}')
    print(f'test_loader batches: {len(test_loader)}')

    print(next(iter(train_loader))[0].shape)