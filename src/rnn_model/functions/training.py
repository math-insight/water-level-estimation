import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    model.train()
    epoch_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(1)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def test_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device
) -> float:
    model.eval()
    epoch_loss = 0
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_transformer_step(
        model,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim,
        device: torch.device
) -> float:
    model.train()
    epoch_loss = 0
    for x, y, time_features, binary_mask in dataloader:
        x, y, time_features, binary_mask = x.to(device), y.to(device), time_features.to(device), binary_mask.to(device)
        y = y.unsqueeze(1)
        outputs = model(x, time_features, binary_mask)
        loss = loss_fn(outputs.loss, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def test_transformer_step(
        model,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device
) -> float:
    model.eval()
    epoch_loss = 0
    with torch.inference_mode():
        for x, y, time_features, binary_mask in dataloader:
            x, y, time_features, binary_mask = x.to(device), y.to(device), time_features.to(device), binary_mask.to(device)
            y = y.unsqueeze(1)
            outputs = model(x, time_features, binary_mask)
            loss = loss_fn(outputs, y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)