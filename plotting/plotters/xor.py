from dataclasses import replace

import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width

from plotting.analysis.xor import XORResultsStore
from plotting.plotters.plotter_base import run_plots
from plotting.plot_specs.xor import XORAxDetailsStore, XORElemDetailsStore
from plotting.utils import setup_axis, plot_line


class XORPlotter:
    def __init__(self):
        self.results = XORResultsStore()
        self.ax_details = XORAxDetailsStore()
        self.elem_details = XORElemDetailsStore()

    def plot_event_rate(self, ax, model_type, phase_mode, ax_name='event_rate', show_y_label=False):
        time, event_rate_mean, event_rate_std = self.results.get_data(model_type=model_type, phase_mode=phase_mode,
                                                                      data_type='output_event_rate')

        time -= time[0]
        half_bin_offset = 0.2
        time += half_bin_offset

        event_line_metadata = self.elem_details.get('event_rate')

        event_rate_line = plot_line(ax, time, event_rate_mean, yerr=event_rate_std, **event_line_metadata.to_kwargs())

        # ax.legend(handles=[event_rate_line], loc=0)

        ax_metadata = self.ax_details.get(ax_name)
        if not show_y_label:
            ax_metadata = replace(ax_metadata, y_label="")

        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_burst_probability(self, ax, model_type, phase_mode, ax_name='burst_probability', show_y_label=False):
        time, bp_mean, bp_std, baseline_bp_mean, baseline_bp_std = self.results.get_data(model_type=model_type,
                                                                                         phase_mode=phase_mode,
                                                                                         data_type='output_burst_probability')

        bp_mean_pct = bp_mean * 100.0
        bp_std_pct = bp_std * 100.0
        baseline_bp_mean_pct = baseline_bp_mean * 100.0
        baseline_bp_std_pct = baseline_bp_std * 100.0

        time -= time[0]
        half_bin_offset = 0.2
        time += half_bin_offset

        bp_line_metadata = self.elem_details.get('burst_probability')

        if model_type == 'burstprop':
            baseline_bp_line_metadata = self.elem_details.get('ma_burst_probability')
        elif model_type == 'burstccn':
            baseline_bp_line_metadata = self.elem_details.get('baseline_burst_probability')
        else:
            raise ValueError(f"Invalid model type {model_type}")

        bp_line = plot_line(ax, time, bp_mean_pct, yerr=bp_std_pct, **bp_line_metadata.to_kwargs())
        baseline_bp_line = plot_line(ax, time, baseline_bp_mean_pct, yerr=baseline_bp_std_pct, **baseline_bp_line_metadata.to_kwargs())

        # if model_type == 'burstprop' and phase_mode == "two_phase":
        if phase_mode == "two_phase":
            ax.legend(handles=[bp_line, baseline_bp_line], loc='lower center')

        ax_metadata = self.ax_details.get(ax_name)
        if not show_y_label:
            ax_metadata = replace(ax_metadata, y_label="")

        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_weight_changes(self, ax, model_type, phase_mode, ax_name='weight_changes', show_y_label=False):
        time, delta_w1, delta_w2 = self.results.get_data(model_type=model_type,
                                                         phase_mode=phase_mode,
                                                         data_type='weight_changes')

        scaling_factor = 1000
        scaled_delta_w1 = delta_w1 * scaling_factor
        scaled_delta_w2 = delta_w2 * scaling_factor

        time -= time[0]
        half_bin_offset = 0.2
        time += half_bin_offset

        hidden1_line_metadata = self.elem_details.get('hidden1_delta_W')
        hidden2_line_metadata = self.elem_details.get('hidden2_delta_W')

        hidden1_line = plot_line(ax, time, scaled_delta_w1, **hidden1_line_metadata.to_kwargs())
        hidden2_line = plot_line(ax, time, scaled_delta_w2, **hidden2_line_metadata.to_kwargs())

        ax.axhline(y=0, color='black', linestyle='--', linewidth=hidden1_line.get_linewidth() * 0.8, zorder=-1)

        if model_type == 'burstprop' and phase_mode == "two_phase":
            ax.legend(handles=[hidden1_line, hidden2_line], loc='lower center')

        ax_metadata = self.ax_details.get(ax_name)
        if not show_y_label:
            ax_metadata = replace(ax_metadata, y_label="")

        setup_axis(ax, **ax_metadata.to_kwargs())
        ax.tick_params(axis='x', which='minor', labelsize=7)


class PhasePlotter:
    def plot_phase_signal(
        self,
        ax,
        phase_mode="two_phase",
        signal_type="teaching_signal",
        show_y_label=True,
        show_y_ticklabels=True,
        n_examples=4,
    ):
        xs = []
        ys = []

        for i in range(n_examples):
            xs.extend([i, i + 0.8, i + 0.8, i + 1.0])
            if phase_mode == "two_phase":
                ys.extend([0.0, 0.0, 1.0, 1.0])
            elif phase_mode == "one_phase":
                ys.extend([1.0, 1.0, 1.0, 1.0])
            else:
                raise ValueError(f"Unknown phase_mode: {phase_mode}")

        xs.append(float(n_examples))
        ys.append(0.0 if phase_mode == "two_phase" else 1.0)

        from plotting.plot_specs.plot_specs_base import PlotColours

        if signal_type == "teaching_signal":
            colour = PlotColours.BURST_PROB
            ylabel = "teaching\nsignal"
        elif signal_type == "plasticity":
            colour = "#DE85B0"
            ylabel = "plasticity"
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")

        linewidth = 2
        fontsize = 12

        ax.plot(xs, ys, linewidth=linewidth, color=colour, solid_joinstyle="miter", solid_capstyle="projecting",)

        if phase_mode == "two_phase":
            marker_specs = {
                "teaching_signal": (0.0, 0.8, "prediction"),
                "plasticity": (0.8, 1.0, "teacher"),
            }
            x_start, x_end, label = marker_specs[signal_type]
            marker_linewidth = 0.88
            y_line = -0.55
            y_cap_bottom = -0.67
            y_cap_top = -0.43
            y_text = -0.68
            if signal_type == "plasticity":
                y_line += 0.34
                y_cap_bottom += 0.34
                y_cap_top += 0.34
                y_text += 0.34

            ax.plot(
                [x_start, x_end],
                [y_line, y_line],
                color="black",
                linewidth=marker_linewidth,
                solid_capstyle="butt",
                clip_on=False,
            )
            ax.vlines(
                [x_start, x_end],
                y_cap_bottom,
                y_cap_top,
                color="black",
                linewidth=marker_linewidth,
                clip_on=False,
            )
            ax.text(
                0.5 * (x_start + x_end),
                y_text,
                label,
                ha="center",
                va="top",
                fontsize=10,
                fontfamily="Consolas",
                color="black",
                clip_on=False,
            )

        ax.spines["left"].set_color(colour)
        ax.spines["left"].set_linewidth(linewidth)

        ax.set_xlim(0.0, n_examples + 0.1)
        ax.set_ylim(-0.3, 1.3)
        ax.set_yticks([0.0, 1.0])
        ax.set_xticks([])
        ax.tick_params(axis="y", colors=colour, width=linewidth, length=7)

        if show_y_ticklabels:
            ax.set_yticklabels(["OFF", "ON"], fontweight="bold", fontfamily="Consolas", fontsize=fontsize)
        else:
            ax.tick_params(axis="y", labelleft=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if show_y_label:
            ax.set_ylabel(
                ylabel,
                rotation=0,
                fontweight="bold",
                fontfamily="Consolas",
                fontsize=fontsize,
                labelpad=15,
                color=colour
            )
            ax.yaxis.label.set_horizontalalignment("right")
            ax.yaxis.label.set_verticalalignment("center")
        else:
            ax.set_ylabel("")


if __name__ == "__main__":
    PLOT_REGISTRY = {
        # BurstProp
        "burstprop_two_phase_event_rate": {
            "fn": lambda p, ax, **kw: p.plot_event_rate(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstprop", "phase_mode": "two_phase", "show_y_label": True},
        },
        "burstprop_one_phase_event_rate": {
            "fn": lambda p, ax, **kw: p.plot_event_rate(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstprop", "phase_mode": "one_phase", "show_y_label": False},
        },
        "burstprop_two_phase_burst_probability": {
            "fn": lambda p, ax, **kw: p.plot_burst_probability(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstprop", "phase_mode": "two_phase", "show_y_label": True},
        },
        "burstprop_one_phase_burst_probability": {
            "fn": lambda p, ax, **kw: p.plot_burst_probability(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstprop", "phase_mode": "one_phase", "show_y_label": False},
        },
        "burstprop_two_phase_weight_change": {
            "fn": lambda p, ax, **kw: p.plot_weight_changes(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstprop", "phase_mode": "two_phase", "show_y_label": True},
        },
        "burstprop_one_phase_weight_change": {
            "fn": lambda p, ax, **kw: p.plot_weight_changes(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstprop", "phase_mode": "one_phase", "show_y_label": False},
        },

        # BurstCCN
        "burstccn_two_phase_event_rate": {
            "fn": lambda p, ax, **kw: p.plot_event_rate(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstccn", "phase_mode": "two_phase", "show_y_label": True},
        },
        "burstccn_one_phase_event_rate": {
            "fn": lambda p, ax, **kw: p.plot_event_rate(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstccn", "phase_mode": "one_phase", "show_y_label": False},
        },
        "burstccn_two_phase_burst_probability": {
            "fn": lambda p, ax, **kw: p.plot_burst_probability(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstccn", "phase_mode": "two_phase", "show_y_label": True},
        },
        "burstccn_one_phase_burst_probability": {
            "fn": lambda p, ax, **kw: p.plot_burst_probability(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstccn", "phase_mode": "one_phase", "show_y_label": False},
        },
        "burstccn_two_phase_weight_change": {
            "fn": lambda p, ax, **kw: p.plot_weight_changes(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstccn", "phase_mode": "two_phase", "show_y_label": True},
        },
        "burstccn_one_phase_weight_change": {
            "fn": lambda p, ax, **kw: p.plot_weight_changes(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"model_type": "burstccn", "phase_mode": "one_phase", "show_y_label": False},
        },
    }

    # run_plots(XORPlotter, PLOT_REGISTRY)
    # run_plots(XORPlotter, PLOT_REGISTRY, "burstprop_two_phase_event_rate")

    PLOT_REGISTRY = {
        "two_phase_teaching_signal": {
            "fn": lambda p, ax, **kw: p.plot_phase_signal(ax, **kw),
            "figsize": (5, 1),
            "kwargs": {"phase_mode": "two_phase",
                       "signal_type": "teaching_signal"
                       }
        },
        "two_phase_plasticity": {
            "fn": lambda p, ax, **kw: p.plot_phase_signal(ax, **kw),
            "figsize": (5, 1),
            "kwargs": {"phase_mode": "two_phase",
                       "signal_type": "plasticity"
                       }
        },
    }
    run_plots(PhasePlotter, PLOT_REGISTRY)

