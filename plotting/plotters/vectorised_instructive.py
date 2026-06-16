from pathlib import Path

import numpy as np

from plotting.analysis.vectorized_instructive_signals import (
    VectorizedInstructiveDataResultsStore,
    VectorizedInstructiveModelResultsStore,
)
from plotting.plot_specs.vectorised_instructive import (
    VectorisedInstructiveAxDetailsStore,
    VectorisedInstructiveElemDetailsStore,
)
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis


class VectorisedInstructivePlotter:
    def __init__(self):
        self.data_results = VectorizedInstructiveDataResultsStore()
        self.model_results = VectorizedInstructiveModelResultsStore()
        self.ax_details = VectorisedInstructiveAxDetailsStore()
        self.elem_details = VectorisedInstructiveElemDetailsStore()

    @staticmethod
    def _condition_title(condition):
        title_overrides = {
            "decreasing_error": r"$\downarrow$error",
            "increasing_error": r"$\uparrow$error",
            "positive_target": r"$\downarrow$error",
            "negative_target": r"$\uparrow$error",
        }
        return title_overrides.get(condition, condition.replace("_", " "))

    def _plot_line(
            self,
            ax,
            plot_data,
            title,
            ax_spec_key,
            vline_x=None,
            show_legend=False,
            show_title=False,
    ):
        p_plus_meta = self.elem_details.get("p_plus")
        p_minus_meta = self.elem_details.get("p_minus")

        ax.plot(
            plot_data["x_p_plus"],
            plot_data["y_p_plus"],
            color=p_plus_meta.line_colour,
            label=p_plus_meta.display_name,
        )
        ax.plot(
            plot_data["x_p_minus"],
            plot_data["y_p_minus"],
            color=p_minus_meta.line_colour,
            label=p_minus_meta.display_name,
        )

        if "p_plus_stderr" in plot_data:
            ax.fill_between(
                plot_data["x_p_plus"],
                np.clip(plot_data["y_p_plus"] - plot_data["p_plus_stderr"], 0.0, 1.0),
                np.clip(plot_data["y_p_plus"] + plot_data["p_plus_stderr"], 0.0, 1.0),
                color=p_plus_meta.line_colour,
                alpha=0.2,
            )
        if "p_minus_stderr" in plot_data:
            ax.fill_between(
                plot_data["x_p_minus"],
                np.clip(plot_data["y_p_minus"] - plot_data["p_minus_stderr"], 0.0, 1.0),
                np.clip(plot_data["y_p_minus"] + plot_data["p_minus_stderr"], 0.0, 1.0),
                color=p_minus_meta.line_colour,
                alpha=0.2,
            )

        if show_title:
            ax.set_title(title.replace("_", " "))
        ax.set_xlabel("error")
        ax.set_ylabel("signal")
        if vline_x is not None:
            ax.axvline(vline_x, linestyle="--", linewidth=1.0, color="black")

        if show_legend:
            legend = ax.legend(loc="upper left", handlelength=1.0)
            for line in legend.get_lines():
                line.set_linewidth(3.0)
        setup_axis(ax, **self.ax_details.get(ax_spec_key).to_kwargs())

    def _plot_means_bar(
            self,
            ax,
            plot_data,
            title,
            ax_spec_key,
            show_title=False,
    ):
        p_plus_meta = self.elem_details.get("p_plus")
        p_minus_meta = self.elem_details.get("p_minus")

        x = np.array([1, 2], dtype=float)
        y = np.asarray(plot_data["means"], dtype=float)
        yerr = np.asarray(plot_data["errors"], dtype=float)
        colours = [p_plus_meta.line_colour, p_minus_meta.line_colour]
        labels = plot_data["labels"]

        ax.bar(
            x, y,
            yerr=yerr,
            color=colours,
            width=0.5,
            capsize=0.0,
            ecolor="black",
            edgecolor="black",  # outline colour
            linewidth=0.8  # outline thickness
        )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if show_title:
            ax.set_title(title.replace("_", " "))

        setup_axis(ax, **self.ax_details.get(ax_spec_key).to_kwargs())

    def _plot_dual_means_bar(
            self,
            ax,
            condition_plot_data,
            condition_labels,
            title,
            y_label,
            y_lims=None,
            y_ticks=None,
            show_title=False,
    ):
        p_plus_meta = self.elem_details.get("p_plus")
        p_minus_meta = self.elem_details.get("p_minus")

        bar_pair_gap = 1.0
        x = np.concatenate([
            np.array([1.0, 2.0]) + condition_idx * (2.0 + bar_pair_gap)
            for condition_idx in range(len(condition_plot_data))
        ])
        y = np.concatenate([np.asarray(plot_data["means"], dtype=float) for plot_data in condition_plot_data])
        yerr = np.concatenate([np.asarray(plot_data["errors"], dtype=float) for plot_data in condition_plot_data])
        colours = [p_plus_meta.line_colour, p_minus_meta.line_colour] * len(condition_plot_data)
        condition_label_x = x.reshape(len(condition_plot_data), 2).mean(axis=1)

        ax.bar(
            x, y,
            yerr=yerr,
            color=colours,
            width=0.5,
            capsize=2,
            ecolor="black",
            edgecolor="black",  # outline colour
            linewidth=1.0  # outline thickness
        )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlim(0.5, x[-1] + 0.5)
        if y_lims is not None:
            ax.set_ylim(y_lims)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        ax.set_ylabel(y_label)
        ax.set_xticks(condition_label_x)
        ax.set_xticklabels(condition_labels)
        if show_title:
            ax.set_title(title.replace("_", " "))

    def _get_model_bar_plot_data(
            self,
            condition,
            zscore_burst_probability=False,
            **model_kwargs,
    ):
        neuron_means = self.model_results.calculate_means(
            condition=condition,
            zscore=zscore_burst_probability,
            **model_kwargs,
        )
        p_plus_neuron_means = neuron_means["p_plus"]
        p_minus_neuron_means = neuron_means["p_minus"]

        return {
            "labels": [
                self.elem_details.get("p_plus").display_name,
                self.elem_details.get("p_minus").display_name,
            ],
            "means": np.array([
                float(np.mean(p_plus_neuron_means)),
                float(np.mean(p_minus_neuron_means)),
            ], dtype=float),
            "errors": np.array([
                float(np.std(p_plus_neuron_means, ddof=1) / np.sqrt(max(len(p_plus_neuron_means), 1))),
                float(np.std(p_minus_neuron_means, ddof=1) / np.sqrt(max(len(p_minus_neuron_means), 1))),
            ], dtype=float),
        }

    def plot_data_condition(self, ax, condition, show_legend=False, show_title=True, **data_kwargs):
        plot_data = self.data_results.get_plot_arrays(condition=condition, **data_kwargs)
        self._plot_line(
            ax,
            plot_data,
            title=self._condition_title(condition),
            ax_spec_key="data",
            vline_x=0.0,
            show_legend=show_legend,
            show_title=show_title,
        )

    def plot_model_condition(
            self,
            ax,
            condition,
            show_legend=False,
            show_title=True,
            zscore_burst_probability=False,
            **model_kwargs,
    ):
        plot_data = self.model_results.get_cdf_plot_data(
            condition=condition,
            zscore=zscore_burst_probability,
            **model_kwargs,
        )
        self._plot_line(
            ax,
            plot_data,
            title=self._condition_title(condition),
            ax_spec_key="model_zscore" if zscore_burst_probability else "model",
            vline_x=0.0 if zscore_burst_probability else 0.5,
            show_legend=show_legend,
            show_title=show_title,
        )
        if not zscore_burst_probability:
            ax.set_xlim(0.28, 0.72)
            ax.set_xticks([0.3, 0.5, 0.7])

    def plot_data_means_bar(self, ax, condition, show_legend=False, show_title=False, **data_kwargs):
        plot_data = self.data_results.get_bar_plot_data(condition=condition, **data_kwargs)
        self._plot_means_bar(
            ax,
            plot_data,
            title=f"Paper data mean: {condition}",
            ax_spec_key="data_bar",
            show_title=show_title,
        )

    def plot_data_dual_means_bar(
            self,
            ax,
            conditions=("decreasing_error", "increasing_error"),
            show_legend=False,
            show_title=False,
            **data_kwargs,
    ):
        condition_data = [
            self.data_results.get_bar_plot_data(condition=condition, **data_kwargs)
            for condition in conditions
        ]
        label_overrides = {
            "decreasing_error": r"$\downarrow$error",
            "increasing_error": r"$\uparrow$error",
        }
        condition_labels = [
            label_overrides.get(condition, condition.replace("_", "\n"))
            for condition in conditions
        ]
        self._plot_dual_means_bar(
            ax,
            condition_plot_data=condition_data,
            condition_labels=condition_labels,
            title="Paper data mean: decreasing and increasing error",
            y_label="SD residual (z-score)",
            show_title=show_title,
        )

    def plot_model_means_bar(
            self,
            ax,
            condition,
            show_legend=False,
            show_title=False,
            zscore_burst_probability=False,
            **model_kwargs,
    ):
        plot_data = self._get_model_bar_plot_data(
            condition=condition,
            zscore_burst_probability=zscore_burst_probability,
            **model_kwargs,
        )
        self._plot_means_bar(
            ax,
            plot_data,
            title=f"Model mean: {condition}",
            ax_spec_key="model_bar",
            show_title=show_title,
        )
        if zscore_burst_probability:
            ax.set_ylabel("Burst probability (z-score)")

    def plot_model_dual_means_bar(
            self,
            ax,
            conditions=("positive_target", "negative_target"),
            show_legend=False,
            show_title=False,
            zscore_burst_probability=False,
            **model_kwargs,
    ):
        condition_data = [
            self._get_model_bar_plot_data(
                condition=condition,
                zscore_burst_probability=zscore_burst_probability,
                **model_kwargs,
            )
            for condition in conditions
        ]
        label_overrides = {
            "positive_target": r"$\downarrow$error",
            "negative_target": r"$\uparrow$error",
        }
        condition_labels = [
            label_overrides.get(condition, condition.replace("_", "\n"))
            for condition in conditions
        ]
        if zscore_burst_probability:
            y_label = "Burst probability (z-score)"
        else:
            y_label = r"$\Delta$ Burst probability"

        self._plot_dual_means_bar(
            ax,
            condition_plot_data=condition_data,
            condition_labels=condition_labels,
            title="Model mean: positive and negative target",
            y_label=y_label,
            show_title=show_title,
        )


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "data_decreasing_error": {
            "fn": lambda p, ax: p.plot_data_condition(ax, condition="decreasing_error"),
            "figsize": (3.5, 4.5),
        },
        "data_increasing_error": {
            "fn": lambda p, ax: p.plot_data_condition(ax, condition="increasing_error"),
            "figsize": (3.5, 4.5),
        },
        "model_positive_target": {
            "fn": lambda p, ax: p.plot_model_condition(ax, condition="positive_target"),
            "figsize": (3.5, 4.5),
        },
        "model_negative_target": {
            "fn": lambda p, ax: p.plot_model_condition(ax, condition="negative_target"),
            "figsize": (3.5, 4.5),
        },
        "data_decreasing_error_bar": {
            "fn": lambda p, ax: p.plot_data_means_bar(ax, condition="decreasing_error"),
            "figsize": (2.5, 4.5),
        },
        "data_increasing_error_bar": {
            "fn": lambda p, ax: p.plot_data_means_bar(ax, condition="increasing_error"),
            "figsize": (2.5, 4.5),
        },
        "model_positive_target_bar": {
            "fn": lambda p, ax: p.plot_model_means_bar(ax, condition="positive_target"),
            "figsize": (2.5, 4.5),
        },
        "model_negative_target_bar": {
            "fn": lambda p, ax: p.plot_model_means_bar(ax, condition="negative_target"),
            "figsize": (2.5, 4.5),
        },
        "data_dual_means_bar": {
            "fn": lambda p, ax: p.plot_data_dual_means_bar(ax),
            "figsize": (2.5, 4.5),
        },
        "model_dual_means_bar": {
            "fn": lambda p, ax: p.plot_model_dual_means_bar(ax),
            "figsize": (2.5, 4.5),
        }
    }

    single_panel_dir = Path(__file__).resolve().parent / "single_panels" / "vectorised_instructive"

    run_plots(
        VectorisedInstructivePlotter,
        PLOT_REGISTRY,
        plot_names=["data_decreasing_error", "data_increasing_error"],
        save_dir=single_panel_dir,
    )
    run_plots(
        VectorisedInstructivePlotter,
        PLOT_REGISTRY,
        plot_names=["model_positive_target", "model_negative_target"],
        save_dir=single_panel_dir,
    )
    # run_plots(VectorisedInstructivePlotter, PLOT_REGISTRY, plot_names=["data_decreasing_error_bar", "data_increasing_error_bar"])
    # run_plots(VectorisedInstructivePlotter, PLOT_REGISTRY, plot_names=["model_positive_target_bar", "model_negative_target_bar"])
    run_plots(VectorisedInstructivePlotter, PLOT_REGISTRY, plot_names=["data_dual_means_bar"])
