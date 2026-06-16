from .base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)


class CIFAR10Dataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)


class ImageNetDataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)