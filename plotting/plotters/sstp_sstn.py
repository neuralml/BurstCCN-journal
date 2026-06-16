import numpy as np

from plotting.analysis.sstp_sstn import SSTpSSTnDataResultsStore, SSTpSSTnModelResultsStore
from plotting.plot_specs.sstp_sstn import SSTpSSTnAxDetailsStore, SSTpSSTnElemDetailsStore
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis


class SSTpSSTnPlotter:
    def __init__(self):
        self.data_results = SSTpSSTnDataResultsStore()
        self.model_results = SSTpSSTnModelResultsStore()
        self.ax_details = SSTpSSTnAxDetailsStore()
        self.elem_details = SSTpSSTnElemDetailsStore()

    def _plot_bar_payload(
            self,
            ax,
            bar_data,
            axis_key,
            title,
            draw_zero_line=False,
            show_legend=False,
            show_title=False,
    ):
        sstp_meta = self.elem_details.get("sstp")
        sstn_meta = self.elem_details.get("sstn")
        colors = [sstp_meta.line_colour, sstn_meta.line_colour]

        early_vals = bar_data["early_mean"]
        late_vals = bar_data["late_mean"]
        early_errs = bar_data.get("early_err")
        late_errs = bar_data.get("late_err")

        x = np.arange(len(early_vals))
        bar_w = 0.25

        ax.bar(
            x - bar_w / 2,
            early_vals,
            bar_w,
            yerr=early_errs,
            capsize=4 if early_errs is not None else 0,
            facecolor="none",
            edgecolor=colors,
            linewidth=1.5,
            label="Early",
        )
        ax.bar(
            x + bar_w / 2,
            late_vals,
            bar_w,
            yerr=late_errs,
            capsize=4 if late_errs is not None else 0,
            color=colors,
            edgecolor=colors,
            label="Late",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(bar_data["x_labels"])
        if show_title:
            ax.set_title(title)
        ax.set_xlim([-0.5, len(x) - 0.5])
        if draw_zero_line:
            ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0, zorder=0)
        setup_axis(ax, **self.ax_details.get(axis_key).to_kwargs())

        if show_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.08))

    def plot_data_cue_bars(self, ax, cue_time=-1.0, halfwidth=0.05, show_legend=False, show_title=False):
        bar_data = self.data_results.get_bar_plot_data(bar_type="cue", cue_time=cue_time, halfwidth=halfwidth)
        ref = bar_data.get("reference_sems", {}).get("cue", {})
        if "early" in ref and "late" in ref:
            bar_data = dict(bar_data)
            bar_data["early_err"] = np.asarray(ref["early"], dtype=float)
            bar_data["late_err"] = np.asarray(ref["late"], dtype=float)
        self._plot_bar_payload(ax, bar_data, axis_key="data_cue_bars", title="Cue",
            show_legend=show_legend, show_title=show_title)

    def plot_data_reward_delta_bars(
            self,
            ax,
            t0=0.0,
            t1=1.0,
            halfwidth=0.05,
            show_legend=False,
            show_title=False,
    ):
        bar_data = self.data_results.get_bar_plot_data(
            bar_type="reward_delta",
            t0=t0,
            t1=t1,
            halfwidth=halfwidth,
        )
        self._plot_bar_payload(
            ax,
            bar_data,
            axis_key="data_reward_delta_bars",
            title="Reward",
            draw_zero_line=True,
            show_legend=show_legend,
            show_title=show_title,
        )

    def plot_model_cue_bars(self, ax, show_legend=False, show_title=False, **model_kwargs):
        bar_data = self.model_results.get_bar_plot_data(bar_type="cue", **model_kwargs)
        self._plot_bar_payload(ax, bar_data, axis_key="model_cue_bars", title="Stimulus",
                               show_legend=show_legend, show_title=show_title)

    def plot_model_error_delta_bars(self, ax, show_legend=False, show_title=False, **model_kwargs):
        bar_data = self.model_results.get_bar_plot_data(bar_type="error_delta", **model_kwargs)
        self._plot_bar_payload(
            ax,
            bar_data,
            axis_key="model_error_delta_bars",
            title="Error",
            draw_zero_line=True,
            show_legend=show_legend,
            show_title=show_title,
        )


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "data_cue_bars": {
            "fn": lambda p, ax: p.plot_data_cue_bars(ax),
            "figsize": (3.5, 4.5),
        },
        "data_reward_delta_bars": {
            "fn": lambda p, ax: p.plot_data_reward_delta_bars(ax),
            "figsize": (3.5, 4.5),
        },
        "model_cue_bars": {
            "fn": lambda p, ax: p.plot_model_cue_bars(ax),
            "figsize": (3.5, 4.5),
        },
        "model_error_delta_bars": {
            "fn": lambda p, ax: p.plot_model_error_delta_bars(ax),
            "figsize": (3.5, 4.5),
        },
    }

    run_plots(
        SSTpSSTnPlotter,
        PLOT_REGISTRY,
        plot_names=[
            "data_cue_bars",
            "data_reward_delta_bars",
            "model_cue_bars",
            "model_error_delta_bars",
        ],
    )
