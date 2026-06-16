from dataclasses import replace

from matplotlib import pyplot as plt, cm
import matplotlib.colors as mcolors

import numpy as np

from plotting.analysis.mnist_representation import MNISTRepresentationResultsStore
from plotting.plot_specs.mnist_representation import MNISTRepresentationElemDetailsStore, \
    MNISTRepresentationAxDetailsStore
from plotting.plot_specs.plot_specs_base import PlotLabels
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis, plot_line


class MNISTRepresentationPlotter:
    def __init__(self):
        self.results = MNISTRepresentationResultsStore()

        self.elem_details = MNISTRepresentationElemDetailsStore()
        self.ax_details = MNISTRepresentationAxDetailsStore()

    def plot_MDS(self, ax, ax_name='MDS'):
        # models = ["ann", "np", "burstccn", "burstccn_Y_block_trained", "burstccn_QY_tied", "burstprop", "edn"]
        # models = ["ann", "burstccn", "burstccn_Y_block_trained", "burstccn_QY_tied", "burstprop", "edn"]
        models = ["ann", "burstccn", "burstccn_Y_block_trained", "burstccn_QY_tied"]

        # models = ["ann", "np"]#, "burstccn", "burstccn_Y_block_trained", "burstccn_QY_tied", "burstprop", "edn"]
        layer_indices = list(range(4))
        epoch = 30
        # epoch = 99

        # Build RDM using the unified helper
        # rdm = self.results.rdm_for_models_layers(
        #     models=models,
        #     layer_indices=layer_indices,
        #     epoch=epoch,
        #     data_type="e",
        #     max_per_class=200,
        # )
        #
        # coords = MDS(
        #     n_components=2,
        #     dissimilarity="precomputed",
        #     random_state=0,
        #     eps=1e-9,
        #     normalized_stress=False,
        # ).fit_transform(rdm)

        coords = self.results.mds_for_model_layers(models=models, layer_indices=layer_indices, epoch=epoch,
                                                   data_type="e", max_per_class=200, seeds=[5])

        coords[:, 0] = -coords[:, 0]
        coords[:, 1] = -coords[:, 1]

        n_layers = len(layer_indices)
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        x_pad = 0.02 * x_range if x_range > 0 else 0.02

        for i, model_name in enumerate(models):
            zorder = len(models) - i
            segment = coords[i * n_layers: (i + 1) * n_layers]
            line_meta = self.elem_details.get(model_name)
            # if model_name == 'ann':
            #     line_meta = replace(line_meta, line_style='--')
            # elif model_name == 'burstccn':
            #     line_meta = replace(line_meta, line_style='--')

            # line_meta = replace(line_meta, zorder=10)
            line = plot_line(ax, segment[:, 0], segment[:, 1], **line_meta.to_kwargs())

            for (x, y, layer_idx) in zip(segment[:, 0], segment[:, 1], layer_indices):
                if layer_idx <= 0:
                    label = 'inputs' if i == 0 else ''
                else:
                    label = str(layer_idx)

                ax.annotate(label, (x + x_pad, y), color=line.get_color(), zorder=100)

        ax_meta = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_meta.to_kwargs())

        ax.legend(loc="lower right", bbox_to_anchor=(1.5, 0.1))

    def plot_TSNE(self, ax, model_name='ann', show_legend=False, colour_by='labels'):
        ref_model_name = 'ann'

        # data_type = 'e'
        layer_idx = 2
        # epoch = 30
        seed=2
        epoch = 99
        max_per_class = 200
        model_kwargs = dict(layer_idx=layer_idx, epoch=epoch, seed=seed,
                            max_per_class=max_per_class, random_state=101)

        tsne_data = self.results.tsne_for_model(data_type='e', model_name=model_name, ref_model_name=ref_model_name, **model_kwargs)

        if colour_by == 'labels':
            target_labels = self.results.get_data(
                model_name=ref_model_name,
                epoch=epoch,
                data_type="targets",
                max_per_class=200,
                seed=seed
            )

            for digit in range(10):
                idx = np.where(target_labels == digit)[0]
                ax.scatter(tsne_data[idx, 0], tsne_data[idx, 1], s=2, label=str(digit))

            elem_details = self.elem_details.get(model_name)
            ax.set_title(elem_details.display_name, fontfamily="Consolas", color="black", fontsize=15, pad=6)

        elif colour_by == 'burst_prob':
            event_rates = self.results.get_data(model_name, data_type='e', **model_kwargs)
            burst_rates = self.results.get_data(model_name, data_type='b_t', **model_kwargs)
            burst_probs = burst_rates / event_rates * 100
            burst_prob_change_mag = np.abs(burst_probs - 50).mean(axis=1)

            norm = mcolors.Normalize(vmin=np.min(burst_prob_change_mag), vmax=np.max(burst_prob_change_mag))
            cmap = cm.get_cmap('viridis')

            ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=2, c=burst_prob_change_mag, cmap=cmap, norm=norm,)

            fig = ax.figure
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(PlotLabels.BURST_PROB_CHANGE_MAGNITUDE)

        if show_legend:
            ax.legend(
                title="Digit",
                bbox_to_anchor=(0.98, 0.5),
                loc="center left",
                borderaxespad=0.0,
                markerscale=3,
            )

        ax_meta = self.ax_details.get('TSNE')
        setup_axis(ax, **ax_meta.to_kwargs())


if __name__ == "__main__":
    REPRESENTATION_PLOT_REGISTRY = {
        "MDS": {
            "fn": lambda p, ax, **kw: p.plot_MDS(ax, **kw),
            "figsize": (5, 4),
        },
        "TSNE_ann": {
            "fn": lambda p, ax, **kw: p.plot_TSNE(ax, **kw),
            "figsize": (3, 3),
            "kwargs": {"model_name": "ann"},
        },
        "TSNE_burstccn": {
            "fn": lambda p, ax, **kw: p.plot_TSNE(ax, **kw),
            "figsize": (3, 3),
            "kwargs": {"model_name": "burstccn"},
        },
        "TSNE_burstccn_Y_block_trained": {
            "fn": lambda p, ax, **kw: p.plot_TSNE(ax, **kw),
            "figsize": (3, 3),
            "kwargs": {"model_name": "burstccn_Y_block_trained"},
        },
        "TSNE_burstccn_QY_tied": {
            "fn": lambda p, ax, **kw: p.plot_TSNE(ax, **kw),
            "figsize": (3, 3),
            "kwargs": {"model_name": "burstccn_QY_tied"},
        },
    }

    run_plots(MNISTRepresentationPlotter, REPRESENTATION_PLOT_REGISTRY, "MDS")
    # run_plots(MNISTRepresentationPlotter, REPRESENTATION_PLOT_REGISTRY, "TSNE_ann")
    # run_plots(MNISTRepresentationPlotter, REPRESENTATION_PLOT_REGISTRY, "TSNE_burstccn")
    # run_plots(MNISTRepresentationPlotter, REPRESENTATION_PLOT_REGISTRY, "TSNE_burstccn_Y_block_trained")
    # run_plots(MNISTRepresentationPlotter, REPRESENTATION_PLOT_REGISTRY, "TSNE_burstccn_QY_tied")
