import numpy as np
import torch
import os
import pickle
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, IterableDataset


def get_rgb_data_mean_std(dataset):
    """Compute mean and std for dataset normalization."""
    data_r = np.dstack([dataset[i][0][:, :, 0] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][0][:, :, 1] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][0][:, :, 2] for i in range(len(dataset))])

    mean = np.mean(data_r) / 255., np.mean(data_g) / 255., np.mean(data_b) / 255.
    std = np.std(data_r) / 255., np.std(data_g) / 255., np.std(data_b) / 255.

    return mean, std


def filter_dataset_by_classes(dataset, subset_classes):
    """Filters a dataset to include only specific classes."""
    filtered_data = [data for data in dataset if data[1] in subset_classes]
    filtered_targets = torch.tensor([data[1] for data in dataset if data[1] in subset_classes])
    return TensorDataset(torch.stack([d[0] for d in filtered_data]), filtered_targets)


def get_cached_image_folder(dataset_path, cache_file='dataset_cache.pkl', transform=None):
    """
    Get a cached ImageFolder dataset to avoid re-indexing large dataset on every run.

    Args:
        dataset_path (str): Path to the dataset directory.
        cache_file (str): Path to the cache file for storing the indexed dataset.
        transform (callable, optional): A function/transform to apply to the images.

    Returns:
        ImageFolder: Cached or newly indexed ImageFolder dataset.
    """
    if os.path.exists(cache_file):
        print(f"Loading dataset index from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        # Update the transform dynamically in case it was not saved
        if transform is not None:
            dataset.transform = transform
    else:
        print(f"Indexing dataset at {dataset_path}. This may take a while...")
        dataset = ImageFolder(dataset_path, transform=transform)
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset index cached at: {cache_file}")

    return dataset


def get_dataloader_length(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        if hasattr(dataset, "length"):
            return dataset.length
    return len(dataloader)


class LengthAwareDataset(IterableDataset):
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length


def get_dataloader_kwargs(config):
    dataloader_args = {'num_workers': config.num_workers,
                       'pin_memory': config.pin_memory}

    if config.num_workers != 0: dataloader_args['prefetch_factor'] = config.prefetch_factor
    return dataloader_args
