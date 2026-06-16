import matplotlib.patches as mpatches

from plotting.analysis.sst_density import SSTDensityDataResultsStore
from plotting.plot_specs.plot_specs_base import PlotColours
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis


class SSTDensityPlotter:
    def __init__(self):
        self.data_results = SSTDensityDataResultsStore()

    def plot_brain_area_density(self, ax):
        labels, means, errors = self.data_results.get_sst_density_data(sorted=True)

        ax.errorbar(
            means,
            range(len(means)),
            xerr=errors,
            marker='o',
            linestyle='none',
            color=PlotColours.DEFAULT,
            ecolor=PlotColours.DEFAULT,
            lw=1.2,
            markersize=3.5
        )

        ax.set_yticks(ticks=range(len(labels)))
        ax.set_yticklabels(labels)

        color_map = {
            'medial_lateral': ['AIp', 'TEa', 'ECT', 'PERI', 'AIv', 'GU', 'VISC', 'AId', 'ILA', 'PL', 'ORBm'],
            'association': ['RSPagl', 'AUDv', 'VISpm', 'VISam', 'ORBvl', 'RSPv', 'ACAv', 'PTLp', 'VISpl',
                            'ACAd', 'VISl', 'VISp', 'AUDp', 'AUDd', 'RSPd', 'VISal', 'ORBl'],
        }
        for label in ax.get_yticklabels():
            name = label.get_text()
            if name in color_map['medial_lateral']:
                label.set_color('#ff1e4f')
            elif name in color_map['association']:
                label.set_color('#ff8742')
            else:
                label.set_color('#159c76')
            label.set_size(8)

        patches = [
            mpatches.Patch(color='#ff1e4f', label='Medial prefrontal\nand lateral areas'),
            mpatches.Patch(color='#ff8742', label='Medial associations\nand audio/visual areas'),
            mpatches.Patch(color='#159c76', label='Motor and somato-\nsensory areas')
        ]
        # ax.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.3, 0.0), fontsize='small')
        ax.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.15, 0.00), fontsize='small')
        # ax.set_title("Data")
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)

        ax.set_ylim(-1, len(labels))


        setup_axis(ax, x_label="Mean SST density\n(cells per $mm^3$)", y_label="Brain area")


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "brain_area_density": {
            "fn": lambda p, ax: p.plot_brain_area_density(ax),
            "figsize": (4, 6.5),
        },
    }

    run_plots(SSTDensityPlotter, PLOT_REGISTRY, plot_names="brain_area_density")
