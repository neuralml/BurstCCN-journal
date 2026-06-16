import numpy as np
from matplotlib import colors as mcolors

from plotting.analysis.dalean_ablation import DaleanAblationResultsStore
from plotting.plot_specs.dalean_ablation import DaleanAblationAxDetailsStore, DaleanAblationElemDetailsStore
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis


class DaleanAblationPlotter:
    def __init__(self):
        self.results = DaleanAblationResultsStore()
        self.ax_details = DaleanAblationAxDetailsStore()
        self.elem_details = DaleanAblationElemDetailsStore()

    @staticmethod
    def _darken_color(color, factor=0.65):
        rgb = np.array(mcolors.to_rgb(color), dtype=float)
        return tuple(np.clip(rgb * factor, 0.0, 1.0))

    def _get_source_data(self, source, metric="relative_weight"):
        if source == "data":
            return self.results.get_ablation_data(), "ablation_data"
        if source == "model":
            if metric == "relative_weight":
                return self.results.get_ablation_model_metric(metric), "ablation_model"
            if metric == "event_rate_change":
                return self.results.get_ablation_model_metric(metric), "ablation_model_event_rate_change"
            if metric == "burst_probability_change":
                return self.results.get_ablation_model_metric(metric), "ablation_model_burst_probability_change"
            raise ValueError(f"Unknown model ablation metric: {metric}")
        raise ValueError(f"Unknown source: {source}")

    def plot_ablation(self, ax, source="data", plot_type="bar", show_legend=False, metric="relative_weight"):
        values_dict, ax_name = self._get_source_data(source, metric=metric)

        labels = list(values_dict.keys())
        x = np.arange(len(labels))
        width = 0.5

        display_names = [self.elem_details.get(label).display_name or label for label in labels]
        bar_colours = [self.elem_details.get(label).line_colour or "gray" for label in labels]

        if plot_type == "box":
            data = [values_dict.get(label, []) for label in labels]
            bp = ax.boxplot(
                data,
                positions=x,
                widths=width,
                patch_artist=True,
                showfliers=False,
            )

            for box, color in zip(bp["boxes"], bar_colours):
                box.set_facecolor(color)
                box.set_edgecolor("black")
                box.set_linewidth(1.5)

            for key in ("whiskers", "caps", "medians"):
                for artist in bp[key]:
                    artist.set_color("black")
                    artist.set_linewidth(1.5)
        elif plot_type == "bar":
            y_lists = [values_dict.get(label, []) for label in labels]
            mean_values = [np.nan if len(values) == 0 else np.mean(values) for values in y_lists]
            sem_values = [
                0 if len(values) == 0 else np.std(values, ddof=1) / np.sqrt(len(values))
                for values in y_lists
            ]

            bars = ax.bar(x, mean_values, width, color=bar_colours, edgecolor="black", linewidth=1)
            ax.errorbar(x, mean_values, yerr=sem_values, color="black", fmt="none", capsize=3, linewidth=1.5, zorder=6)

            if show_legend:
                ax.legend(bars, display_names, loc=self.ax_details.get(ax_name).legend_location or "best")
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        rng = np.random.default_rng(seed=0)
        for idx, label in enumerate(labels):
            y_vals = values_dict.get(label, [])
            if len(y_vals) == 0:
                continue
            jittered_x = x[idx] + rng.normal(0, 0.05, size=len(y_vals))
            ax.scatter(
                jittered_x,
                y_vals,
                color=self._darken_color(bar_colours[idx], factor=0.65),
                s=10,
                zorder=5,
                alpha=1.0,
            )

        ax.axhline(0, color="black")
        ax.set_xticks(x)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        setup_axis(ax, **self.ax_details.get(ax_name).to_kwargs())
        ax.set_xticklabels(display_names, rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2)

    def plot_ablation_data(self, ax, plot_type="bar", show_legend=False):
        self.plot_ablation(ax, source="data", plot_type=plot_type, show_legend=show_legend)

    def plot_ablation_model(self, ax, plot_type="bar", show_legend=False, metric="relative_weight"):
        self.plot_ablation(
            ax,
            source="model",
            plot_type=plot_type,
            show_legend=show_legend,
            metric=metric,
        )

    def plot_ablation_model_event_rate_change(self, ax, plot_type="bar", show_legend=False):
        self.plot_ablation_model(
            ax,
            plot_type=plot_type,
            show_legend=show_legend,
            metric="event_rate_change",
        )

    def plot_ablation_model_burst_probability_change(self, ax, plot_type="bar", show_legend=False):
        self.plot_ablation_model(
            ax,
            plot_type=plot_type,
            show_legend=show_legend,
            metric="burst_probability_change",
        )


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "ablation_data": {
            "fn": lambda p, ax: p.plot_ablation_data(ax),
            "figsize": (4, 3),
        },
        "ablation_model": {
            "fn": lambda p, ax: p.plot_ablation_model(ax),
            "figsize": (4, 3),
        },
        "ablation_model_event_rate_change": {
            "fn": lambda p, ax: p.plot_ablation_model_event_rate_change(ax),
            "figsize": (4, 3),
        },
        "ablation_model_burst_probability_change": {
            "fn": lambda p, ax: p.plot_ablation_model_burst_probability_change(ax),
            "figsize": (4, 3),
        },
    }

    run_plots(DaleanAblationPlotter, PLOT_REGISTRY, plot_names=["ablation_data", "ablation_model"])
