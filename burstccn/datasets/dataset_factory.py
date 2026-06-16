import io
import json
import os
import platform
import sys
from pathlib import Path

from functools import partial

import torch
import torchvision
from PIL import Image

from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
from torchvision.transforms import transforms

import webdataset as wds

from burstccn.datasets.dataset_utils import LengthAwareDataset, get_cached_image_folder, get_dataloader_kwargs
from burstccn.datasets.image_datasets import MNISTDataset, CIFAR10Dataset


def log_and_continue(exn):
    print(f"[WebDataset Decoding Error] {repr(exn)}")
    return True  # continue after error


def decode_sample(sample, synset_to_idx):
    image = Image.open(io.BytesIO(sample['jpg'])).convert("RGB")
    label_str = sample['cls'].decode('utf-8')
    label = torch.tensor(synset_to_idx[label_str], dtype=torch.long)
    return image, label


def identity(x):
    return x


class DatasetFactory:
    @classmethod
    def get_dataset(cls, config, train_batch_size, test_batch_size):
        dataset_name = config.dataset_name

        if dataset_name == "mnist":
            return cls._get_mnist(config, train_batch_size=train_batch_size,
                                  test_batch_size=test_batch_size)
        elif dataset_name == "fmnist":
            return cls._get_fmnist(config, train_batch_size=train_batch_size,
                                   test_batch_size=test_batch_size)
        elif dataset_name == "cifar10":
            return cls._get_cifar10(config, train_batch_size=train_batch_size,
                                    test_batch_size=test_batch_size)
        elif dataset_name == "imagenet":
            return cls._get_imagenet_imagefolder(config, train_batch_size=train_batch_size,
                                                 test_batch_size=test_batch_size)
        elif dataset_name == 'imagenet_web':
            return cls._get_imagenet_webdataset(config, train_batch_size=train_batch_size,
                                                test_batch_size=test_batch_size)
        elif dataset_name == 'simple_increasing_task':
            return cls._get_simple_increasing_task(config, train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    @classmethod
    def _get_mnist(cls, config, train_batch_size, test_batch_size):
        train_transforms = [torchvision.transforms.ToTensor()]
        test_transforms = [torchvision.transforms.ToTensor()]

        if config.standardise_inputs:
            # Append Normalize transform if standardisation is requested
            train_transforms.append(torchvision.transforms.Normalize([0.131], [0.308]))
            test_transforms.append(torchvision.transforms.Normalize([0.131], [0.308]))

        train_transform = torchvision.transforms.Compose(train_transforms)
        test_transform = torchvision.transforms.Compose(test_transforms)

        full_train_dataset = torchvision.datasets.MNIST(config.root, train=True, download=True,
                                                        transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(config.root, train=False, download=True, transform=test_transform)

        if config.use_validation:
            train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000])
            val_dataset.dataset.transform = test_transform
        else:
            train_dataset, val_dataset = full_train_dataset, None

        train_dataset = MNISTDataset(train_dataset)
        val_dataset = MNISTDataset(val_dataset) if val_dataset else None
        test_dataset = MNISTDataset(test_dataset)

        dataloader_kwargs = get_dataloader_kwargs(config)

        train_loader = train_dataset.get_dataloader(batch_size=train_batch_size, shuffle=True, **dataloader_kwargs)
        val_loader = val_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False,
                                                **dataloader_kwargs) if val_dataset else None
        test_loader = test_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False, **dataloader_kwargs)

        return train_loader, val_loader, test_loader

    @classmethod
    def _get_fmnist(cls, config, train_batch_size, test_batch_size):
        train_transforms = [torchvision.transforms.ToTensor()]
        test_transforms = [torchvision.transforms.ToTensor()]

        if config.standardise_inputs:
            # Append Normalize transform if standardisation is requested
            train_transforms.append(torchvision.transforms.Normalize([0.2860], [0.3530]))
            test_transforms.append(torchvision.transforms.Normalize([0.2860], [0.3530]))

        train_transform = torchvision.transforms.Compose(train_transforms)
        test_transform = torchvision.transforms.Compose(test_transforms)

        full_train_dataset = torchvision.datasets.FashionMNIST(config.root, train=True, download=True,
                                                               transform=train_transform)
        test_dataset = torchvision.datasets.FashionMNIST(config.root, train=False, download=True,
                                                         transform=test_transform)

        if config.use_validation:
            train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000])
            val_dataset.dataset.transform = test_transform
        else:
            train_dataset, val_dataset = full_train_dataset, None

        train_dataset = MNISTDataset(train_dataset)
        val_dataset = MNISTDataset(val_dataset) if val_dataset else None
        test_dataset = MNISTDataset(test_dataset)

        dataloader_kwargs = get_dataloader_kwargs(config)

        train_loader = train_dataset.get_dataloader(batch_size=train_batch_size, shuffle=True, **dataloader_kwargs)
        val_loader = val_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False,
                                                **dataloader_kwargs) if val_dataset else None
        test_loader = test_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False, **dataloader_kwargs)

        return train_loader, val_loader, test_loader

    @classmethod
    def _get_cifar10(cls, config, train_batch_size, test_batch_size):
        train_transforms = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]

        test_transforms = [torchvision.transforms.ToTensor()]

        if config.standardise_inputs:
            # Append Normalize transform if standardisation is requested
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            train_transforms.append(torchvision.transforms.Normalize(mean, std))
            test_transforms.append(torchvision.transforms.Normalize(mean, std))

        train_transform = torchvision.transforms.Compose(train_transforms)
        test_transform = torchvision.transforms.Compose(test_transforms)

        base_train_dataset = torchvision.datasets.CIFAR10(config.root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(config.root, train=False, download=True, transform=test_transform)

        if config.use_validation:
            base_val_dataset = torchvision.datasets.CIFAR10(config.root, train=True, download=True, transform=test_transform)

            n_total = len(base_train_dataset)  # 50_000
            n_train = 45_000
            n_val = n_total - n_train
            gen = torch.Generator().manual_seed(getattr(config, "split_seed", 0))  # deterministic split

            train_indices, val_indices = random_split(range(len(base_train_dataset)), [n_train, n_val], generator=gen)

            train_dataset = Subset(base_train_dataset, train_indices)
            val_dataset = Subset(base_val_dataset, val_indices)

            # train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val], generator=gen)
            # val_dataset.dataset.transform = test_transform
        else:
            train_dataset, val_dataset = base_train_dataset, None

        train_dataset = CIFAR10Dataset(train_dataset)
        val_dataset = CIFAR10Dataset(val_dataset) if val_dataset else None
        test_dataset = CIFAR10Dataset(test_dataset)

        dataloader_kwargs = get_dataloader_kwargs(config)

        train_loader = train_dataset.get_dataloader(batch_size=train_batch_size, shuffle=True, **dataloader_kwargs)
        val_loader = val_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False,
                                                **dataloader_kwargs) if val_dataset else None
        test_loader = test_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False, **dataloader_kwargs)

        return train_loader, val_loader, test_loader

    @classmethod
    def _get_imagenet_imagefolder(cls, config, train_batch_size, test_batch_size):
        """Loads ImageNet and properly splits into training, validation, and test sets."""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dir = os.path.join(config.root, 'train')
        test_dir = os.path.join(config.root, 'val')
        train_cache_file = os.path.join(config.root, 'imagenet_train_cache.pkl')
        test_cache_file = os.path.join(config.root, 'imagenet_val_cache.pkl')

        train_dataset = get_cached_image_folder(train_dir, cache_file=train_cache_file, transform=train_transform)
        val_dataset = get_cached_image_folder(test_dir, cache_file=test_cache_file,
                                              transform=test_transform) if config.use_validation else None
        test_dataset = get_cached_image_folder(test_dir, cache_file=test_cache_file, transform=test_transform)

        # train_dataset = ImageNetDataset(train_dataset)
        # val_dataset = ImageNetDataset(val_dataset) if val_dataset else None
        # test_dataset = ImageNetDataset(test_dataset)

        # dataloader_kwargs = {'num_workers': kwargs['num_workers'],
        #                      'pin_memory': True,
        #                      'prefetch_factor': kwargs['prefetch_factor']}

        # train_loader = train_dataset.get_dataloader(batch_size=train_batch_size, shuffle=True, **dataloader_kwargs)
        # val_loader = val_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False, **dataloader_kwargs) if val_dataset else None
        # test_loader = test_dataset.get_dataloader(batch_size=test_batch_size, shuffle=False, **dataloader_kwargs)

        dataloader_kwargs = get_dataloader_kwargs(config)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
                                  **dataloader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False,
                                **dataloader_kwargs) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False,
                                 **dataloader_kwargs)

        return train_loader, val_loader, test_loader

    @classmethod
    def _get_imagenet_webdataset(cls, config, train_batch_size, test_batch_size):
        """Loads ImageNet WebDataset shards with optional validation split (Windows-safe)."""

        # Root paths
        root_path = Path(config.root).resolve()
        train_dir = root_path / "train"
        val_dir = root_path / "val"

        synset_path = root_path.parent / "imagenet_class_index.json"

        # Load synset mapping
        with open(synset_path) as f:
            synset_to_idx = {v[0]: int(k) for k, v in json.load(f).items()}

        # Shard discovery

        train_shards = sorted(train_dir.glob("train-*.tar"))
        val_shards = sorted(val_dir.glob("val-*.tar"))

        # Cross-platform binary streaming
        if platform.system() == "Windows":
            cmd = f"{sys.executable} -c \"import sys; sys.stdout.buffer.write(open(sys.argv[1], 'rb').read())\" 2>NUL"
        else:
            cmd = "cat"

        train_urls = [f"pipe:{cmd} {p.as_posix()}" for p in train_shards]
        # train_urls = [f"pipe:{cmd} {p.as_posix()}" for p in train_shards[1200:]]
        val_urls = [f"pipe:{cmd} {p.as_posix()}" for p in val_shards]

        # Transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        num_train_batches = 1281167 // train_batch_size
        # num_train_batches = 81167 // train_batch_size
        num_test_batches = 50000 // test_batch_size

        # Train loader
        train_dataset = (
            wds.WebDataset(train_urls, shardshuffle=True, handler=log_and_continue)
            .shuffle(10000)
            .map(partial(decode_sample, synset_to_idx=synset_to_idx))
            .map_tuple(train_transform, identity)
            .batched(train_batch_size, partial=False)
        )

        dataloader_kwargs = get_dataloader_kwargs(config)

        train_dataset = LengthAwareDataset(train_dataset, num_train_batches)
        train_loader = DataLoader(train_dataset, batch_size=None, **dataloader_kwargs)

        # Validation loader (optional)
        val_loader = None
        if config.use_validation:
            val_dataset = (
                wds.WebDataset(val_urls, handler=log_and_continue)
                .map(partial(decode_sample, synset_to_idx=synset_to_idx))
                .map_tuple(test_transform, identity)
                .batched(test_batch_size, partial=False)
            )
            val_dataset = LengthAwareDataset(val_dataset, num_test_batches)
            val_loader = DataLoader(val_dataset, batch_size=None, **dataloader_kwargs)

        # Test loader
        test_dataset = (
            wds.WebDataset(val_urls, handler=log_and_continue)
            .map(partial(decode_sample, synset_to_idx=synset_to_idx))
            .map_tuple(test_transform, identity)
            .batched(test_batch_size, partial=False)
        )

        test_dataset = LengthAwareDataset(test_dataset, num_test_batches)

        test_loader = DataLoader(test_dataset, batch_size=None, **dataloader_kwargs)

        return train_loader, val_loader, test_loader

    # @classmethod
    # def _get_simple_increasing_task(cls, config, train_batch_size, test_batch_size):
    #     assert config.use_validation is False
    #
    #     # Create linearly spaced data points between 0 and 1
    #     num_train = config.num_train_samples if hasattr(config, 'num_train_samples') else 1000
    #     num_test = config.num_test_samples if hasattr(config, 'num_test_samples') else 200
    #
    #     x_train = torch.linspace(0, 1, num_train).unsqueeze(1)
    #     y_train = x_train.clone()
    #
    #     x_test = torch.linspace(0, 1, num_test).unsqueeze(1)
    #     y_test = x_test.clone()
    #
    #     train_dataset = TensorDataset(x_train, y_train)
    #     test_dataset = TensorDataset(x_test, y_test)
    #
    #     dataloader_kwargs = get_dataloader_kwargs(config)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **dataloader_kwargs)
    #     val_loader = None  # No validation set for this simple task
    #     test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **dataloader_kwargs)
    #
    #     return train_loader, val_loader, test_loader

    @classmethod
    def _get_simple_increasing_task(cls, config, train_batch_size, test_batch_size, seq_len=10,
                                    input_value=1.0, target_value=0.5):
        assert config.use_validation is False

        input_value = getattr(config, "input_value", input_value)
        target_value = getattr(config, "target_value", target_value)

        num_train = getattr(config, 'num_train_samples', 1)
        num_test = getattr(config, 'num_test_samples', 1)

        class ConstantInputDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, input_value=1.0, target_value=0.5):
                self.inputs = torch.full((num_samples, 1), input_value)
                self.targets = torch.full((num_samples, 1), target_value)

            def set_input_value(self, new_input):
                self.inputs.fill_(new_input)

            def set_target_value(self, new_target):
                self.targets.fill_(new_target)

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]

        # Instantiate datasets
        train_dataset = ConstantInputDataset(num_train, input_value=input_value, target_value=target_value)
        test_dataset = ConstantInputDataset(num_test, input_value=input_value, target_value=target_value)

        # Optionally keep a reference for external use:
        # cls.constant_train_dataset = train_dataset
        # cls.constant_test_dataset = test_dataset

        # Build loaders
        dataloader_kwargs = get_dataloader_kwargs(config)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **dataloader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **dataloader_kwargs)

        val_loader = None  # No validation set for this task

        return train_loader, val_loader, test_loader


import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs/dataset", config_name="simple_increasing_task", version_base=None)
def main(cfg: DictConfig):
    print("Loading dataset with config:")
    print(cfg)

    train_loader, _, _ = DatasetFactory.get_dataset(config=cfg, train_batch_size=1, test_batch_size=1)

    # Example: show one batch
    batch = next(iter(train_loader))
    x, y = batch
    print("Batch shape:", x.shape, y.shape)
    print("x:", x)
    print("y:", y)


if __name__ == "__main__":
    main()

