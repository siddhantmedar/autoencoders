#!/usr/bin/env python3
"""
Training script for AutoEncoder
Includes training loop, testing, and checkpoint management

Usage:
    # Train with default settings
    python train.py --train

    # Train with custom hyperparameters
    python train.py --train --epochs 50 --lr 1e-3 --batch_size 128

    # Train with weight decay
    python train.py --train --epochs 100 --lr 5e-4 --weight_decay 1e-4

    # Test model (loads best.pt by default)
    python train.py

    # Custom save directory
    python train.py --train --save_dir my_checkpoints

Arguments:
    --train         Enable training mode (default: test mode)
    --epochs        Number of training epochs (default: 100)
    --batch_size    Batch size for training (default: 64)
    --lr            Learning rate (default: 5e-4)
    --weight_decay  Weight decay for regularization (default: 0.0)
    --num_workers   Number of data loading workers (default: 4)
    --save_dir      Directory to save checkpoints (default: checkpoints)
"""

import os
import json
from datetime import datetime
from pathlib import Path
import argparse
import tomllib

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import AutoEncoder
from dataset import get_dataloaders


def train(
    model,
    train_loader,
    val_loader,
    optimizer="adam",
    weight_decay=0.0,
    learning_rate=5e-4,
    epochs=100,
    save_path="checkpoints",
    device=None,
):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    loss_fn = nn.MSELoss()

    opt = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs", timestamp))

    print(f"Training on device: {device}")
    print(f"Optimizer: {optimizer}, LR: {learning_rate}, WD: {weight_decay}")
    print("-" * 60)

    best_val_loss = float("inf")

    # Training loop with progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch"
        )
        for x, _ in train_pbar:
            x = x.view(x.size(0), -1).to(device)

            opt.zero_grad()
            _, _, reconstructed = model(x)
            loss = loss_fn(reconstructed, x)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            num_batches += 1

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= num_batches

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.view(x.size(0), -1).to(device)

                _, _, reconstructed = model(x)
                loss = loss_fn(reconstructed, x)

                val_loss += loss.item()
                num_batches += 1

        val_loss /= num_batches

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        current_lr = opt.param_groups[0]["lr"]
        writer.add_scalar("Learning_rate", current_lr, epoch)

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{current_lr:.6f}",
            }
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_path, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            tqdm.write(f"  → Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            last_path = os.path.join(save_path, "last.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                },
                last_path,
            )

    # Save final checkpoint
    final_path = os.path.join(save_path, "last.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epochs - 1,
        },
        final_path,
    )

    writer.close()

    print(f"\n{'-' * 60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return best_val_loss


def test(model, test_loader, checkpoint_path=None, device=None):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully")

    print(f"Testing on device: {device}")
    print("-" * 60)

    # Test
    model.eval()
    test_loss = 0.0
    num_batches = 0

    loss_fn = nn.MSELoss()
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for x, _ in test_pbar:
            x = x.view(x.size(0), -1).to(device)

            _, _, reconstructed = model(x)
            loss = loss_fn(reconstructed, x)

            test_loss += loss.item()
            num_batches += 1

            test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    test_loss /= num_batches

    print(f"\n{'-' * 60}")
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print("-" * 60)

    results = {"test_loss": test_loss}

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"checkpoints/test_results_{timestamp}.json"
    os.makedirs("checkpoints", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train AutoEncoder on MNIST")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--train", action="store_true", help="Enable training mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    CONFIG_PATH = Path(__file__).parent / "config.toml"
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    d_in = cfg["model"]["d_in"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    num_workers = args.num_workers or cfg["training"]["num_workers"]
    epochs = args.epochs or cfg["training"]["num_epochs"]
    learning_rate = args.lr
    weight_decay = args.weight_decay
    train_mode = args.train

    model = AutoEncoder(d_in=d_in)
    train_loader = get_dataloaders(split="train", batch_size=batch_size, num_workers=num_workers)
    val_loader = get_dataloaders(split="val", batch_size=batch_size, num_workers=num_workers)

    if train_mode:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer="adam",
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            epochs=epochs,
            save_path=args.save_dir,
        )
    else:
        test_loader = get_dataloaders(split="test", batch_size=batch_size, num_workers=num_workers)
        test(model=model, test_loader=test_loader, checkpoint_path="checkpoints/best.pt")
