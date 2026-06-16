import os

from burstccn.datasets.dataset_factory import DatasetFactory
import torch
from tqdm import tqdm

from burstccn.datasets.dataset_utils import get_dataloader_length


def compare_imagenet_decoding():
    root_imagefolder = "E:\\CLS-LOC"
    root_webdataset = "E:\\imagenet\\webdataset"

    train_batch_size = 128  # Any batch size
    test_batch_size = 1000
    use_validation = True

    print("Loading DataLoaders...")
    train_loader_if, val_loader_if, test_loader_if = DatasetFactory._get_imagenet_imagefolder(
        root=root_imagefolder,
        use_validation=use_validation,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=4
    )

    train_loader_wds, val_loader_wds, test_loader_wds = DatasetFactory._get_imagenet_webdataset(
        root=root_webdataset,
        use_validation=use_validation,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=4
    )

    # loader_if = val_loader_if if use_validation else test_loader_if
    # loader_wds = val_loader_wds if use_validation else test_loader_wds

    loader_if = train_loader_if
    loader_wds = train_loader_wds

    print(f"\nDataset sizes (DataLoader length):")
    print(f"ImageFolder - Train: {get_dataloader_length(train_loader_if)}, Val: {get_dataloader_length(val_loader_if) if val_loader_if else 'None'}, Test: {get_dataloader_length(test_loader_if)}")
    print(f"WebDataset  - Train: {get_dataloader_length(train_loader_wds)}, Val: {get_dataloader_length(val_loader_wds) if val_loader_wds else 'None'}, Test: {get_dataloader_length(test_loader_wds)}")

    # ---- WebDataset pass ----
    print("\nChecking WebDataset decoding...")
    stats_wds = []
    labels_wds = []
    n_samples_wds = 0

    for imgs, labels in tqdm(loader_wds, desc="WebDataset"):
        batch_size_actual = imgs.size(0)
        if (batch_size_actual != train_batch_size): print(f"{batch_size_actual} != {train_batch_size}")
        for i in range(batch_size_actual):
            img = imgs[i].unsqueeze(0)
            label = labels[i]

            stats_wds.append({
                "shape": tuple(img.shape),
                "dtype": str(img.dtype),
                "min": img.min().item(),
                "max": img.max().item(),
                "mean": img.mean().item(),
                "std": img.std().item(),
            })
            labels_wds.append(label.item())
            n_samples_wds += 1

    print(f"\nTotal WebDataset samples: {n_samples_wds}")

    # ---- ImageFolder pass ----
    print("\nChecking ImageFolder decoding...")
    stats_if = []
    labels_if = []
    n_samples_if = 0

    for imgs, labels in tqdm(loader_if, desc="ImageFolder"):
        batch_size_actual = imgs.size(0)
        if (batch_size_actual != train_batch_size): print(f"{batch_size_actual} != {train_batch_size}")
        for i in range(batch_size_actual):
            img = imgs[i].unsqueeze(0)
            label = labels[i]

            stats_if.append({
                "shape": tuple(img.shape),
                "dtype": str(img.dtype),
                "min": img.min().item(),
                "max": img.max().item(),
                "mean": img.mean().item(),
                "std": img.std().item(),
            })
            labels_if.append(label.item())
            n_samples_if += 1

    print(f"\nTotal ImageFolder samples: {n_samples_if}")

    # --- Print summary ---
    print("\nWebDataset decoding stats (first 5 samples):")
    for s in stats_wds[:5]:
        print(s)

    print("\nImageFolder decoding stats (first 5 samples):")
    for s in stats_if[:5]:
        print(s)

    mean_wds = torch.tensor([s["mean"] for s in stats_wds]).mean()
    std_wds = torch.tensor([s["std"] for s in stats_wds]).mean()
    mean_if = torch.tensor([s["mean"] for s in stats_if]).mean()
    std_if = torch.tensor([s["std"] for s in stats_if]).mean()

    print(f"\nAverage Mean: WebDataset={mean_wds:.4f}, ImageFolder={mean_if:.4f}")
    print(f"Average Std:  WebDataset={std_wds:.4f}, ImageFolder={std_if:.4f}")

    # --- Label stats ---
    labels_if_set = set(labels_if)
    labels_wds_set = set(labels_wds)

    print(f"\nLabel range WebDataset : min={min(labels_wds)}, max={max(labels_wds)}, unique={len(labels_wds_set)}")
    print(f"Label range ImageFolder: min={min(labels_if)}, max={max(labels_if)}, unique={len(labels_if_set)}")

    common = labels_if_set & labels_wds_set
    print(f"Common labels: {len(common)} out of {len(labels_if_set)} (ImageFolder) and {len(labels_wds_set)} (WebDataset)")


# import sys
# import platform
# from pathlib import Path
# import webdataset as wds
#
# def count_webdataset_samples(shard_dir):
#     root_path = Path(shard_dir).resolve()
#     shards = sorted(root_path.glob("val-*.tar"))  # adjust for train if needed
#
#     if platform.system() == "Windows":
#         cmd = f"{sys.executable} -c \"import sys; sys.stdout.buffer.write(open(sys.argv[1], 'rb').read())\""
#     else:
#         cmd = "cat"
#
#     urls = [f"pipe:{cmd} {p.as_posix()}" for p in shards]
#
#     # IMPORTANT: Do not decode → just iterate samples
#     dataset = wds.WebDataset(urls)
#
#     count = 0
#     for sample in dataset:
#         count += 1
#         if count % 10000 == 0:
#             print(f"Counted {count} samples...")
#     print(f"\nTotal samples in {shard_dir}: {count}")
#     return count

if __name__ == "__main__":
    # torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 1
    # os.environ["PYTHONWARNINGS"] = "default"

    # val_dir = "E:\\imagenet\\webdataset\\val"
    # count_webdataset_samples(val_dir)
    compare_imagenet_decoding()
