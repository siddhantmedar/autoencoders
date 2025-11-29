import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import AutoEncoder
from config import cfg
from data import fetch_dataset
from evaluate import validate

from datetime import datetime
import os
os.makedirs('./checkpoints', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model,train_loader,optimizer,criterion):
    model.train()
    running_loss = 0
    
    for idx,(inputs,_) in enumerate(train_loader):
        inputs = inputs.view(inputs.size(0),-1)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/autoencoder_{timestamp}')

    model = AutoEncoder(cfg.d_in)
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    criterion = nn.MSELoss()
    train_loader = fetch_dataset(split="train")

    model = model.to(device)
    criterion = criterion.to(device)
    
    best_val_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        epoch_loss = train_one_epoch(model,train_loader,optimizer,criterion)
        print(f"Epoch {epoch+1}, Average Train Loss: {epoch_loss:.4f}")

        # evaluate the model after every epoch
        avg_loss = validate(model,criterion)
        print(f"Epoch {epoch+1}, Average Validation Loss: {avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(model.state_dict(), './checkpoints/best_ae_model.pth')
            print(f"Model saved with val loss: {avg_loss:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', avg_loss, epoch)

    writer.close()


if __name__ == "__main__":
    train()