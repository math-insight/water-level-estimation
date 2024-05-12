import matplotlib.pyplot as plt
from .datasets import WLEDataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


def plot_losses(train_losses: list, test_losses: list, epochs_list: list) -> None:
    plt.plot(epochs_list, train_losses, label="Train loss")
    plt.plot(epochs_list, test_losses, label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    max_value = max(max(train_losses), max(test_losses))
    plt.ylim(0, max_value + 0.1 * max_value)
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def eval_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
    model.eval()
    residuals = []
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1)
            y_pred = model(x)
            residuals.extend((y_pred.view(-1) - y.view(-1)).tolist())
    residuals = np.array(residuals)

    print_residuals_stats(residuals)
    plot_residuals(residuals)


def plot_residuals(residuals: np.ndarray) -> None:
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(np.arange(len(residuals)), residuals, s=1)
    plt.title("Residuals over Predictions")
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")
    max_residual = max(abs(residuals))
    plt.ylim(-max_residual, max_residual)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.1)

    plt.subplot(1, 2, 2)

    hist, bins = np.histogram(residuals, bins=100, density=True)
    x = (bins[:-1] + bins[1:]) / 2
    plt.plot(x, hist)
    plt.ylabel("Density")
    max_bin = max(abs(bins))
    plt.xlim(-max_bin, max_bin)
    plt.ylim(0)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.1)
    plt.title("Residuals density")
    plt.show()


def print_residuals_stats(residuals: np.ndarray, precision: int = 5) -> None:
    print(f"mean: {np.mean(residuals):>8.{precision}f}")
    print(f"std: {np.std(residuals):>9.{precision}f}")
    print(f"min: {np.min(residuals):>9.{precision}f}")
    print(f"Q1: {np.percentile(residuals, q=25):>10.{precision}f}")
    print(f"Q2: {np.percentile(residuals, q=50):>10.{precision}f}")
    print(f"Q3: {np.percentile(residuals, q=75):>10.{precision}f}")
    print(f"max: {np.max(residuals):>9.{precision}f}")


def plot_predictions(
        model,
        data,
        seq_len,
        x_scaler,
        y_scaler,
        device,
        dataset_class=WLEDataSet
):
    if not is_normalized(data):
        data.iloc[:, 1] = y_scaler.transform(data.iloc[:, 1].values.reshape(-1, 1))
        data.iloc[:, 2:] = x_scaler.transform(data.iloc[:, 2:])

    dataset = dataset_class(
        dataframe=data,
        sequence_length=seq_len
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1
    )

    model.eval()
    y_true = []
    y_preds = []
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1)
            y_pred = model(x)
            y_true.append(y.squeeze().item())
            y_preds.append(y_pred.squeeze().item())

    y_true = np.array(y_true).reshape(-1, 1)
    y_preds = np.array(y_preds).reshape(-1, 1)

    y_true = y_scaler.inverse_transform(y_true)
    y_preds = y_scaler.inverse_transform(y_preds)

    plt.figure(figsize=(20, 8))
    plt.plot(list(range(len(y_preds))), y_true, label="True value", linewidth=0.5)
    plt.plot(list(range(len(y_preds))), y_preds, label="Prediction", linewidth=0.5)
    plt.legend(loc="best")
    plt.xlim(0, len(y_preds))
    plt.show()


def is_normalized(df: pd.DataFrame) -> bool:
    return ((df.iloc[:, 1:] >= 0) & (df.iloc[:, 1:] <= 1)).all().all()