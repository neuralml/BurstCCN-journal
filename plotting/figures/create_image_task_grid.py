from collections import defaultdict
import json
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import random
from typing import List, Tuple

from burstccn.datasets.dataset_utils import get_cached_image_folder
from PIL import Image


def is_image_folder_like(dataset):
    return hasattr(dataset, "samples")


def load_imagenet_class_metadata(*candidate_paths):
    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            class_index = json.load(f)
        synset_to_name = {
            synset: class_name.replace("_", " ")
            for synset, class_name in class_index.values()
        }
        name_to_synset = {
            class_name.replace("_", " "): synset
            for synset, class_name in class_index.values()
        }
        return synset_to_name, name_to_synset
    return {}, {}


def resolve_class_names(class_names, class_name_map=None):
    if class_name_map is None:
        class_name_map = {}
    return [class_name_map.get(class_name, class_name) for class_name in class_names]


def collect_examples_by_class_fast(
    dataset,
    examples_per_class: int,
    num_classes_to_show: int,
    transform,
    seed: int = 42,
    selected_classes=None,
    selected_example_indices_by_class=None,
) -> Tuple[dict, List[str]]:
    """
    Efficiently collect image examples using dataset.samples (ImageFolder-style).
    """
    random.seed(seed)
    class_to_paths = defaultdict(list)
    for path, label in dataset.samples:
        class_to_paths[label].append(path)

    if selected_classes is None:
        available_classes = sorted(class_to_paths.keys())
        selected_classes = random.sample(available_classes, num_classes_to_show)

    class_to_images = {}
    for idx in selected_classes:
        selected_indices = (
            selected_example_indices_by_class.get(idx)
            if selected_example_indices_by_class is not None
            else None
        )
        if selected_indices:
            paths = [class_to_paths[idx][i] for i in selected_indices]
        else:
            paths = random.sample(class_to_paths[idx], examples_per_class)
        images = [transform(Image.open(p).convert("RGB")) for p in paths]
        class_to_images[idx] = images

    class_names = [dataset.classes[idx] for idx in selected_classes]
    return class_to_images, class_names


def collect_examples_by_class_cifar(
    dataset,
    examples_per_class: int,
    num_classes_to_show: int,
    seed: int = 42,
    selected_classes=None,
    selected_example_indices_by_class=None,
) -> Tuple[dict, List[str]]:
    """
    Collect examples from in-memory dataset like CIFAR-10.
    """
    random.seed(seed)

    if selected_classes is None:
        all_classes = sorted(set(label for _, label in dataset))
        selected_classes = random.sample(all_classes, num_classes_to_show)

    class_to_examples = defaultdict(list)
    for img, label in dataset:
        if label in selected_classes:
            class_to_examples[label].append(img)

    class_to_images = {}
    for cls in selected_classes:
        selected_indices = (
            selected_example_indices_by_class.get(cls)
            if selected_example_indices_by_class is not None
            else None
        )
        if selected_indices:
            sampled_examples = [class_to_examples[cls][i] for i in selected_indices]
        else:
            sampled_examples = random.sample(class_to_examples[cls], examples_per_class)
        class_to_images[cls] = sampled_examples

    class_names = [dataset.classes[i] for i in selected_classes]
    return class_to_images, class_names


def collect_examples_by_class_auto(
    dataset,
    examples_per_class: int,
    num_classes_to_show: int,
    transform,
    seed: int = 42,
    selected_classes=None,
    selected_example_indices_by_class=None,
) -> Tuple[dict, List[str]]:
    """
    Automatically choose fast or in-memory strategy based on dataset type.
    """
    if is_image_folder_like(dataset):
        return collect_examples_by_class_fast(
            dataset,
            examples_per_class,
            num_classes_to_show,
            transform,
            seed,
            selected_classes,
            selected_example_indices_by_class,
        )
    else:
        return collect_examples_by_class_cifar(
            dataset,
            examples_per_class,
            num_classes_to_show,
            seed,
            selected_classes,
            selected_example_indices_by_class,
        )


def collect_indexed_examples_by_class_fast(
    dataset,
    max_examples_per_class,
    transform,
    selected_classes,
) -> Tuple[dict, List[str]]:
    class_to_paths = defaultdict(list)
    for path, label in dataset.samples:
        class_to_paths[label].append(path)

    class_to_images = {}
    for cls in selected_classes:
        indexed_images = []
        for index, path in enumerate(class_to_paths[cls][:max_examples_per_class]):
            indexed_images.append((index, transform(Image.open(path).convert("RGB"))))
        class_to_images[cls] = indexed_images

    class_names = [dataset.classes[i] for i in selected_classes]
    return class_to_images, class_names


def collect_indexed_examples_by_class_cifar(
    dataset,
    max_examples_per_class,
    selected_classes,
) -> Tuple[dict, List[str]]:
    class_to_examples = defaultdict(list)
    for img, label in dataset:
        if label in selected_classes:
            class_to_examples[label].append(img)

    class_to_images = {}
    for cls in selected_classes:
        class_to_images[cls] = list(enumerate(class_to_examples[cls][:max_examples_per_class]))

    class_names = [dataset.classes[i] for i in selected_classes]
    return class_to_images, class_names


def collect_indexed_examples_by_class_auto(
    dataset,
    max_examples_per_class,
    transform,
    selected_classes,
) -> Tuple[dict, List[str]]:
    if is_image_folder_like(dataset):
        return collect_indexed_examples_by_class_fast(
            dataset, max_examples_per_class, transform, selected_classes
        )
    return collect_indexed_examples_by_class_cifar(
        dataset, max_examples_per_class, selected_classes
    )


def plot_class_grid(
    class_to_images,
    class_names,
    examples_per_class,
    dataset_name="Dataset",
    save_path="figure_pdfs/grid.pdf",
    max_visible_classes=4,
    figsize=(4.0, 4.0),
    show=True,
):
    """
    Plot image examples grouped by class, with classes as columns.
    """
    total_classes = len(class_names)
    classes_to_show = min(max_visible_classes, total_classes)

    fig, axes = plt.subplots(
        examples_per_class,
        classes_to_show,
        figsize=figsize,
        squeeze=False,
    )
    fig.suptitle(
        dataset_name,
        fontsize=24,
        y=1.0,
        fontfamily="Consolas",
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.08,
        top=0.91,
        wspace=0.05,
        hspace=0.05,
    )

    class_keys = list(class_to_images.keys())

    def render_column(col_idx, label, images):
        for row_idx in range(examples_per_class):
            ax = axes[row_idx, col_idx]
            img = images[row_idx]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            ax.imshow(img, interpolation='nearest')
            ax.set_aspect('equal')
            ax.axis('off')
            # if row_idx == examples_per_class - 1:
            #     ax.text(
            #         0.5,
            #         -0.08,
            #         label,
            #         transform=ax.transAxes,
            #         ha="center",
            #         va="top",
            #         fontsize=8,
            #         fontfamily="Consolas",
            #         fontweight="bold",
            #     )

    for col in range(classes_to_show):
        class_idx = class_keys[col]
        class_name = class_names[col]
        render_column(col, class_name, class_to_images[class_idx])

    save_path = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    print(f"Saved grid to {save_path}")


def plot_class_selection_sheets(
    class_to_images,
    class_names,
    examples_per_class=50,
    dataset_name="Dataset",
    save_dir="pdf_resources/image_tasks/selection_sheets",
    cols=10,
    figsize=(12.0, 7.0),
    show=True,
):
    save_dir = os.path.join(os.path.dirname(__file__), save_dir)
    os.makedirs(save_dir, exist_ok=True)
    rows = (examples_per_class + cols - 1) // cols

    for class_key, class_name in zip(class_to_images.keys(), class_names):
        images = class_to_images[class_key]
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        fig.suptitle(
            f"{dataset_name}: {class_name}",
            fontsize=20,
            y=0.96,
            fontfamily="Consolas",
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.88, wspace=0.12, hspace=0.22)

        for cell_idx, ax in enumerate(axes.flat):
            ax.axis("off")
            if cell_idx >= examples_per_class or cell_idx >= len(images):
                continue

            actual_index, img = images[cell_idx]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            ax.imshow(img, interpolation="nearest")
            ax.text(
                0.02,
                0.98,
                str(actual_index),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontfamily="Consolas",
                fontweight="bold",
                color="white",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.7, pad=1.0),
            )

        safe_class_name = class_name.lower().replace(" ", "_").replace("/", "_")
        save_path = os.path.join(save_dir, f"{dataset_name.lower()}_{safe_class_name}_selection.pdf")
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        print(f"Saved selection sheet to {save_path}")


# ==== Main Entry Point ====
if __name__ == "__main__":
    CREATE_SELECTION_SHEETS = False

    # dataset_name = "ImageNet"  # or "ImageNet"
    dataset_names = ["CIFAR-10", "ImageNet"]
    for dataset_name in dataset_names:
        class_name_map = None
        selected_classes = None
        selected_example_indices_by_class = None

        if dataset_name == "CIFAR-10":
            seed = 5
            examples_per_class = 3
            num_classes_to_show = 3
            figsize = (3.09, 3.71)
            transform = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
            ])
            dataset = torchvision.datasets.CIFAR10(
                root='../../data', train=True, download=True, transform=transform
            )
            selected_class_names = ["bird", "frog", "truck"]
            selected_classes = [dataset.class_to_idx[name] for name in selected_class_names]
            selected_example_indices_by_class = {
                dataset.class_to_idx["bird"]: [40, 8, 20],
                dataset.class_to_idx["frog"]: [1, 4, 24],
                dataset.class_to_idx["truck"]: [7, 28, 21],
            }

        elif dataset_name == "ImageNet":
            seed = 3
            examples_per_class = 3
            num_classes_to_show = 4
            figsize = (4.83, 3.67)
            root = r"E:\CLS-LOC"
            split = "val"
            data_dir = os.path.join(root, split)
            cache_file = os.path.join(root, f"imagenet_{split}_cache.pkl")

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

            dataset = get_cached_image_folder(data_dir, cache_file=cache_file, transform=transform)
            class_name_map, name_to_synset = load_imagenet_class_metadata(
                os.path.join(root, "imagenet_class_index.json"),
                os.path.join(os.path.dirname(root), "imagenet_class_index.json"),
                r"E:\imagenet\imagenet_class_index.json",
            )
            selected_class_names = ["African elephant", "tractor", "strawberry", "volcano"]
            selected_classes = [
                dataset.class_to_idx[name_to_synset[name]]
                for name in selected_class_names
            ]
            selected_example_indices_by_class = {
                dataset.class_to_idx[name_to_synset["African elephant"]]: [43, 1, 12],
                dataset.class_to_idx[name_to_synset["tractor"]]: [12, 3, 8],
                dataset.class_to_idx[name_to_synset["strawberry"]]: [0, 14, 34],
                dataset.class_to_idx[name_to_synset["volcano"]]: [8, 48, 11],
            }

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        class_to_images, class_names = collect_examples_by_class_auto(
            dataset,
            examples_per_class,
            num_classes_to_show,
            transform,
            seed=seed,
            selected_classes=selected_classes,
            selected_example_indices_by_class=selected_example_indices_by_class,
        )
        class_names = resolve_class_names(class_names, class_name_map)

        plot_class_grid(
            class_to_images,
            class_names,
            examples_per_class,
            dataset_name=dataset_name,
            save_path=f"pdf_resources/image_tasks/{dataset_name.lower()}_grid.pdf",
            max_visible_classes=num_classes_to_show,
            figsize=figsize,
        )

        if CREATE_SELECTION_SHEETS:
            selection_class_to_images, selection_class_names = collect_indexed_examples_by_class_auto(
                dataset,
                max_examples_per_class=50,
                transform=transform,
                selected_classes=selected_classes,
            )
            selection_class_names = resolve_class_names(selection_class_names, class_name_map)
            plot_class_selection_sheets(
                selection_class_to_images,
                selection_class_names,
                examples_per_class=50,
                dataset_name=dataset_name,
                show=True,
            )
