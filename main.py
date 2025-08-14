import os
import torch
from torch.utils.data import DataLoader
from dataload import get_dataloader
from trajectory_model import LSTMModel
from test_utils import autoregressive_predict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters
input_size = 3
hidden_size = 128
num_layers = 2
output_size = 3
num_epochs = 50
lr = 1e-4
stride = 60
seq_len = 120
pred_len = 400
model_path = "trained_model_3.pt"
base_path = "15"

print("Loading Data...")
train_loader, valid_loader, test_loader = get_dataloader(base_path=base_path, seq_len=seq_len, stride=stride)

print("Defining Model...")
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

def train_model(model, train_loader, valid_loader, num_epochs=num_epochs, lr=lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for x_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for x_val, y_val in tqdm(valid_loader, desc="Validation", leave=False):
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred = model(x_val)
                loss = criterion(y_pred, y_val)
                total_valid_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Train Loss: {avg_train_loss} | Valid Loss: {avg_valid_loss}")

        torch.save(model.state_dict(), model_path)
        print(f"Model has saved: {model_path}")

train_model(model, train_loader, valid_loader)