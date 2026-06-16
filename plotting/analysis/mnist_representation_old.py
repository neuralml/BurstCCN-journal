from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE, MDS

from plotting.analysis.results_store_base import ResultsStore, file_cache_decorator


class MNISTRepresentationResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path=f'mnist_representation.pkl')

        self.base_path = Path(r"D:\various backups\burst_models\poster_figures\journal\activity_data")

        self._balanced_subset_indices_cache = {}

        self.results_map = {
            'fa': 'burstccn_4h_fa_exact_best_no_mom',
            'fa_burstccn': 'burstccn_4h_fa_best_no_mom',
            'fa_burstccn_noQ': 'burstccn_4h_fa_noQ_best_no_mom',
            'fa_burstprop': 'burstprop_4h_fa_best_no_mom',
            'fa_burstprop_exact': 'burstprop_4h_fa_exact_best_no_mom',
            'fa_edn_all_tied': 'edn_4h_fa_sym_no_mom',
            'fa_edn_fb_sym': 'edn_4h_fa_no_mom',
            'fa_edn': 'edn_4h_fa_all_learning_no_mom',
            'node_perturb': 'node_perturb_4h'
        }

    def get_results_folder(self, model_name, seed):
        run_name = self.results_map[model_name]
        results_folder = self.base_path / f'mnist_{run_name}_seed{seed}'
        return results_folder

    def get_balanced_subset_indices(self, model_name, seed, max_per_class=200):
        key = (model_name, seed, max_per_class)
        if key not in self._balanced_subset_indices_cache:
            targets = self.get_target_labels(model_name, seed, max_per_class=None)
            target_sort_indices = []
            for i in range(10):
                target_sort_indices += [np.where(targets == i)[0][:max_per_class]]
            target_sort_indices = np.concatenate(target_sort_indices)
            self._balanced_subset_indices_cache[key] = target_sort_indices

        return self._balanced_subset_indices_cache[key]

    def get_target_labels(self, model_name, seed, max_per_class=None):
        results_folder = self.get_results_folder(model_name, seed)
        file_name = results_folder / 'target_labels_0.npy'
        target_labels = np.load(file_name)

        if max_per_class is not None:
            subset_indices = self.get_balanced_subset_indices(model_name, seed, max_per_class)
            target_labels = target_labels[subset_indices]

        return target_labels

    def get_activity(self, model_name, seed, layer_name, epoch, max_per_class=None, burst_data=False):
        results_folder = self.get_results_folder(model_name, seed)
        file_name = results_folder / f"{layer_name}{'_b' if burst_data else ''}_{epoch}.npy"
        activity = np.load(file_name)

        if max_per_class is not None:
            subset_indices = self.get_balanced_subset_indices(model_name, seed, max_per_class)
            activity = activity[subset_indices]

        if activity.ndim >= 2:
            activity = activity.reshape(activity.shape[0], -1)

        return activity

    def get_burst_prob(self, model_name, seed, layer_name, epoch, max_per_class=None):
        event_rate = self.get_activity(model_name, seed, layer_name, epoch, max_per_class, burst_data=False)
        burst_rate = self.get_activity(model_name, seed, layer_name, epoch, max_per_class, burst_data=True)
        return burst_rate / event_rate

    def get_data(self, model_name, seed, data_type, **kwargs):
        if data_type == 'target_labels':
            return self.get_target_labels(model_name, seed, **kwargs)
        elif data_type == 'event_rate':
            return self.get_activity(model_name, seed, kwargs['layer_name'], kwargs['epoch'], kwargs.get('max_per_class'))
        elif data_type == 'burst_rate':
            return self.get_activity(model_name, seed, kwargs['layer_name'], kwargs['epoch'], kwargs.get('max_per_class'), burst_data=True)
        elif data_type == 'burst_prob':
            return self.get_burst_prob(model_name, seed, kwargs['layer_name'], kwargs['epoch'], kwargs.get('max_per_class'))
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

    def _get_rdv(self, model_name, seeds, data_type, layer_name, epoch, max_per_class=200):
        if model_name == 'node_perturb':
            epoch = 100

        rdvs = []
        for seed in seeds:
            act = self.get_data(model_name, seed, data_type, layer_name=layer_name, epoch=epoch, max_per_class=max_per_class)
            rdvs.append(pdist(act, 'correlation'))

        return np.mean(rdvs, axis=0)

    def _get_rdv_list(self, model_name, seeds, data_type, layer_names, epoch):
        return [self._get_rdv(model_name, seeds, data_type, layer, epoch) for layer in layer_names]

    def _combine_all_rdvs(self, models, layer_names, epoch):
        all_rdvs = []
        for model_name, seed_list, data_type in models:
            rdvs = self._get_rdv_list(model_name, seed_list, data_type, layer_names, epoch)
            all_rdvs.append(np.vstack(rdvs))  # shape (num_layers, rdv_length)
        all_rdvs = np.vstack(all_rdvs)  # shape (num_models × num_layers, rdv_length)
        return squareform(pdist(all_rdvs, 'correlation'))

    @file_cache_decorator()
    def get_MDS_coords(self, models, layer_names, epoch):
        rdm = self._combine_all_rdvs(models, layer_names, epoch)

        coords = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=0,
            eps=1e-9,
            normalized_stress=False
        ).fit_transform(rdm)

        return coords

    def get_tsne_coords(self, model_name, ref_model_name, layer_name='fc2', epoch=30, max_per_class=200):
        # Load activity for this model
        activity = self.get_data(model_name, 1, 'event_rate', layer_name=layer_name, epoch=epoch, max_per_class=max_per_class)

        # Always run t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(activity)

        # Align with reference model if not the first
        if model_name != ref_model_name:
            ref_act = self.get_data(ref_model_name, 1, 'event_rate', layer_name=layer_name, epoch=epoch, max_per_class=max_per_class)
            ref_tsne = TSNE(n_components=2, random_state=42).fit_transform(ref_act)
            _, tsne_data, _ = procrustes(ref_tsne, tsne_data)

        return tsne_data


def get_rdv(store, model_name, seeds, data_type, layer_name, epoch, max_per_class=200):
    if model_name == 'node_perturb':
        epoch = 100

    rdvs = []
    for seed in seeds:
        act = store.get_activity(model_name, seed, layer_name, epoch, max_per_class=max_per_class, burst_data=(data_type == DataType.BURSTS))
        rdvs.append(pdist(act, 'correlation'))

    # print(f"{model_name} {layer_name} {epoch}: activity shape = {act.shape}")

    return np.mean(rdvs, axis=0)


def get_rdv_list(store, model_name, seeds, data_type, layer_names, epoch):
    return [get_rdv(store, model_name, seeds, data_type, layer, epoch) for layer in layer_names]

def combine_all_rdvs(store, models, layer_names, epoch):
    all_rdvs = []
    for model_name, seed_list, data_type in models:
        rdvs = get_rdv_list(store, model_name, seed_list, data_type, layer_names, epoch)
        all_rdvs.append(np.vstack(rdvs))  # shape (num_layers, rdv_length)
    all_rdvs = np.vstack(all_rdvs)  # shape (num_models × num_layers, rdv_length)
    return squareform(pdist(all_rdvs, 'correlation'))

def plot_MDS(store, models, layer_names, epoch):
    rdm = combine_all_rdvs(store, models, layer_names, epoch)

    coords = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=0,
        eps=1e-9,
        normalized_stress=False
    ).fit_transform(rdm)

    fig, ax = plt.subplots(figsize=(6, 6))
    n_layers = len(layer_names)
    x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])

    for i, (model_name, _, data_type) in enumerate(models):
        segment = coords[i * n_layers:(i + 1) * n_layers]
        color = get_colour(model_name, data_type)
        line = ax.plot(segment[:, 0], segment[:, 1], 'o-', label=display_name(model_name, data_type), color=color, linewidth=2)
        for x, y, label in zip(segment[:, 0], segment[:, 1], layer_names):
            ax.annotate(label, (x + 0.02 * x_range, y), color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.legend()
    plt.tight_layout()
    plt.show()

class DataType(Enum):
    BURSTS = 'bursts'
    EVENTS = 'events'

def get_colour(model_type, data_type):
    colour_dict = {
        ('fa', DataType.EVENTS): '#000000',
        ('fa_burstccn', DataType.EVENTS): '#6881D8',
        ('fa_burstprop', DataType.EVENTS): '#B84C3E',
        ('fa_edn', DataType.EVENTS): '#50B47B',
        ('node_perturb', DataType.EVENTS): '#A684C7',
    }
    return colour_dict.get((model_type, data_type), '#999999')

def display_name(model_type, data_type):
    display_name_dict = {
        'fa': 'FA',
        'fa_burstccn': 'BurstCCN',
        'fa_burstccn_noQ': 'BurstCCN (fb sym)',
        'fa_burstprop': 'Burstprop',
        'fa_burstprop_exact': 'Backprop (burstprop model)',
        'fa_edn_all_tied': 'EDN (full sym)',
        'fa_edn_fb_sym': 'EDN (fb sym)',
        'fa_edn': 'EDN',
        'node_perturb': 'NP'
    }
    return display_name_dict.get(model_type, model_type) + (' (bursts)' if data_type == DataType.BURSTS else '')



def plot_TSNE_with_labels(results, ax, model_name, show_legend=False, ax_name='tsne'):
    ref_model_name = 'fa'
    max_per_class = 200

    tsne_coords = results.get_tsne_coords(model_name, ref_model_name, max_per_class=max_per_class)
    target_labels = results.get_data(ref_model_name, 1, 'target_labels', max_per_class=max_per_class)
    point_size = 4

    for digit in range(10):
        indices = np.where(target_labels == digit)
        x, y = tsne_coords[indices, 0], tsne_coords[indices, 1]
        ax.scatter(x, y, s=point_size, label=str(digit))

    if show_legend:
        ax.legend(
            title='Digit',
            bbox_to_anchor=(1.02, 0.5),
            loc='center left',
            borderaxespad=0.0
        )

    # ax_meta = self.plot_details.get_ax_metadata(ax_name)


if __name__ == '__main__':
    store = MNISTRepresentationResultsStore()


    ax = plt.subplot(111)
    plot_TSNE_with_labels(store, ax, model_name='fa')
    plt.show()
    import sys
    sys.exit()

    # indices = store.get_balanced_subset_indices('fa', max_per_class=200)
    # print("Subset indices checksum:", hashlib.md5(indices.tobytes()).hexdigest())

    models = [
        ('fa', [1, 2, 3], DataType.EVENTS),
        ('fa_burstccn', [1, 2, 3], DataType.EVENTS),
        ('fa_burstprop', [1, 2, 3], DataType.EVENTS),
        ('fa_edn', [1, 2, 3], DataType.EVENTS),
        ('node_perturb', [1, 2, 3], DataType.EVENTS),
    ]

    layer_names = ['inputs', 'fc1', 'fc2', 'fc3', 'fc4']
    epoch = 30

    plot_MDS(store, models, layer_names, epoch)
