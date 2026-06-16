import torch
from torch.utils.data import Dataset, Subset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_dataloader(self, batch_size, shuffle, **dataloader_kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
