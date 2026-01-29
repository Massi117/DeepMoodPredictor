import numpy as np
import torch
from torch.utils.data import Dataset

class COPEDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get the data and label
        cope_data = self.data[index]
        label = self.target[index]

        # Normalize the data thorugh per-subject z-score normalization
        cope_data = (cope_data - np.mean(cope_data)) / np.std(cope_data)

        # Convert to tensors
        volume = torch.tensor(cope_data, dtype=torch.float32, requires_grad=True).unsqueeze(0)  # (1, 91, 109, 91) expects # of chanennels first
        label = torch.tensor([1] if label == 0 else [0], dtype=torch.float32)

        return volume, label