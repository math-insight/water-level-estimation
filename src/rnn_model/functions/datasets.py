import pandas as pd
from torch.utils.data import Dataset
import torch


class WLEDataSet(Dataset):
    def __init__(self, dataframe: pd.DataFrame, sequence_length: int):
        self.dataframe = dataframe
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataframe) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = torch.tensor(self.dataframe.iloc[idx:idx + self.sequence_length, 2:].values, dtype=torch.float32)
        y = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)

        return x, y


class TransformerWLEDataSet(Dataset):
    def __init__ (self, dataframe: pd.DataFrame, sequence_length: int):
        self.dataframe = dataframe
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataframe) - self.sequence_length + 1

    def __getitem__(self, idx):
        past_values = torch.tensor(self.dataframe.iloc[idx + 1:idx + self.sequence_length + 1, 3:].values, dtype=torch.float32)
        future_values = torch.tensor(self.dataframe.iloc[idx, 3:].values, dtype=torch.float32)
        past_time_features = torch.tensor(self.dataframe.iloc[idx + 1:idx + self.sequence_length + 1, :3].values, dtype=torch.int16)
        future_time_features = torch.tensor(self.dataframe.iloc[idx, :3].values, dtype=torch.int16)
        past_observed_mask = torch.tensor((-self.dataframe.iloc[idx + 1: idx + self.sequence_length, 3:].isna()).values, dtype=torch.bool)

        return past_values, future_values, past_time_features, future_time_features, past_observed_mask