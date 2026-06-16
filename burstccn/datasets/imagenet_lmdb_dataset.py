import lmdb
import pickle
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO


class ImageNetLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None  # defer opening until worker starts

        # Load keys
        with lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False) as env:
            with env.begin() as txn:
                self.keys = pickle.loads(txn.get(b"__keys__"))
                self.class_to_idx = pickle.loads(txn.get(b"__class_to_idx__"))

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False, readahead=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self._init_env()  # create env inside worker
        key = self.keys[index].encode("ascii")
        with self.env.begin() as txn:
            byteflow = txn.get(key)
        img_bytes, label = pickle.loads(byteflow)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
