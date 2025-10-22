import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def to_tensor(x):
    if torch.is_tensor(x):
        return x.float()
    elif hasattr(x, "values"):
        return torch.tensor(x.values, dtype=torch.float32)
    else:
        return torch.tensor(x, dtype=torch.float32)


def create_dataloaders(tX_user, tX_content, ty, vX_user, vX_content, vy, batch_size):
    train_loader = DataLoader(
        TensorDataset(tX_user, tX_content, ty), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(vX_user, vX_content, vy), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
