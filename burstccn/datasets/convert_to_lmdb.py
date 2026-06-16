import os
import lmdb
import pickle
from tqdm import tqdm
from PIL import Image
import torchvision.datasets as datasets
from multiprocessing import Pool, cpu_count

def read_image(path):
    """Load image as bytes"""
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def process_sample(args):
    index, (path, label) = args
    img_bytes = read_image(path)
    if img_bytes is None:
        return None
    return index, (img_bytes, label)

def build_lmdb(imagefolder_root, lmdb_path, num_workers=8, max_gb=200):
    print(f"Scanning dataset in {imagefolder_root}")
    dataset = datasets.ImageFolder(imagefolder_root)
    samples = dataset.samples
    print(f"Found {len(samples)} samples across {len(dataset.classes)} classes.")

    # Directly set map_size
    map_size = max_gb * (1 << 30)
    print(f"📌 Setting LMDB map_size to: {max_gb} GB\n")

    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, readonly=False, lock=True)

    with Pool(num_workers) as pool:
        with env.begin(write=True) as txn:
            for result in tqdm(
                pool.imap_unordered(process_sample, enumerate(samples)),
                total=len(samples),
                desc="Writing LMDB"
            ):
                if result is None:
                    continue
                index, (img_bytes, label) = result
                key = f"{index:08}".encode("ascii")
                txn.put(key, pickle.dumps((img_bytes, label)))

            # Store metadata
            txn.put(b"__keys__", pickle.dumps([f"{i:08}" for i in range(len(samples))]))
            txn.put(b"__class_to_idx__", pickle.dumps(dataset.class_to_idx))

    env.sync()
    env.close()
    print(f"\n✅ LMDB dataset saved to {lmdb_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Path to ImageNet folder (train/ or val/)")
    parser.add_argument("--out", type=str, required=True, help="Output LMDB file path")
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 16), help="Number of parallel workers")
    parser.add_argument("--max-gb", type=int, default=200, help="Maximum LMDB map size in GB")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_lmdb(args.data_root, args.out, num_workers=args.workers, max_gb=args.max_gb)