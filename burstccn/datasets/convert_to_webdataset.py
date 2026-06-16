import os
import tarfile
from glob import glob
from tqdm import tqdm
import argparse
import io
import math
import random
from multiprocessing import Pool


def write_shard(shard_id, shard_images, output_dir, split):
    shard_path = os.path.join(output_dir, f"{split}-{shard_id:05d}.tar")
    with tarfile.open(shard_path, "w") as tar:
        for i, img_path in enumerate(shard_images):
            class_name = os.path.basename(os.path.dirname(img_path))
            img_key = f"{shard_id:05d}_{i:04d}"
            # Add image
            tar.add(img_path, arcname=f"{img_key}.jpg")
            # Add label
            cls_bytes = class_name.encode("utf-8")
            tarinfo = tarfile.TarInfo(f"{img_key}.cls")
            tarinfo.size = len(cls_bytes)
            tar.addfile(tarinfo, fileobj=io.BytesIO(cls_bytes))
    return shard_path


def process_chunk(args):
    shard_id, shard_images, output_dir, split = args
    return write_shard(shard_id, shard_images, output_dir, split)


def parallel_shard_imagenet(input_dir, output_dir, split, samples_per_shard, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    all_images = glob(os.path.join(input_dir, "*", "*.JPEG"))
    print(f"[{split}] Found {len(all_images)} images.")

    print(f"Shuffling images before creating shards...")
    random.shuffle(all_images)

    chunks = []
    num_shards = math.ceil(len(all_images) / samples_per_shard)
    for shard_id in range(num_shards):
        start = shard_id * samples_per_shard
        end = start + samples_per_shard
        shard_images = all_images[start:end]
        chunks.append((shard_id, shard_images, output_dir, split))

    print(f"Creating {num_shards} shards using {num_workers} workers...")

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_chunk, chunks), total=num_shards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    parallel_shard_imagenet(
        args.data_root,
        args.out,
        args.split,
        args.samples_per_shard,
        args.num_workers,
    )
