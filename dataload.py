import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import OrderedDict

BatchSize = 50
base_path = "15"

class CSVCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        else:
            df = pd.read_csv(path)
            self.cache[path] = df
            self.cache.move_to_end(path)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return df

class TrajectoryDataset(Dataset):
    def __init__(self, folder_path, seq_len=120, stride=60, cache_size=100):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.samples = []
        self.cache = CSVCache(max_size=cache_size)
        self.prepare_samples()

    def prepare_samples(self):
        all_csv_paths = glob.glob(os.path.join(self.folder_path, "*.csv"))
        for file_path in all_csv_paths:
            df = pd.read_csv(file_path)
            if len(df) < self.seq_len + 1:
                continue
            max_start = len(df) - self.seq_len - 1
            for i in range(0, max_start + 1, self.stride):
                self.samples.append((file_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start_idx = self.samples[idx]
        df = self.cache.get(file_path)
        data = df.to_numpy()
        x_seq = data[start_idx:start_idx + self.seq_len]
        y_seq = data[start_idx + 1:start_idx + self.seq_len + 1, 0:3]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

def get_dataloader(base_path=base_path, seq_len=120, stride=60):
    train_dataset = TrajectoryDataset(os.path.join(base_path, "train"), seq_len=seq_len, stride=stride)
    valid_dataset = TrajectoryDataset(os.path.join(base_path, "valid"), seq_len=seq_len, stride=stride)
    test_dataset = TrajectoryDataset(os.path.join(base_path, "test"), seq_len=seq_len, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BatchSize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)

    # Debug: Print Batch Shapes
    """
    print(f"\nTrain Batch Shape (stride={stride}):")
    for x, y in train_loader:
        print("X:", x.shape, "Y:", y.shape)
        break

    print("\nValid Batch Shape:")
    for x, y in valid_loader:
        print("X:", x.shape, "Y:", y.shape)
        break

    print("\nTest Batch Shape:")
    for x, y in test_loader:
        print("X:", x.shape, "Y:", y.shape)
        break
    """

    return train_loader, valid_loader, test_loader