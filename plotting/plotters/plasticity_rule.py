import numpy as np

from plotting.analysis.plasticity_rule import PlasticityRuleResultsStore
from plotting.plot_specs.plasticity_rule import PlasticityRuleAxDetailsStore, PlasticityRuleElemDetailsStore
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis, add_line_aligned_label, plot_line


class PlasticityRulePlotter:
    def __init__(self):
        self.results = PlasticityRuleResultsStore()
        self.ax_details = PlasticityRuleAxDetailsStore()
        self.elem_details = PlasticityRuleElemDetailsStore()

    def plot_delta_W(self, ax, ax_name='delta_W'):
        poisson_rates = self.results.get_data('poisson_rate')
        delta_W, delta_W_sem = self.results.get_data('delta_W')

        poisson_rates = np.concatenate(([0], poisson_rates))
        delta_W = np.concatenate(([0], delta_W))
        delta_W_sem = np.concatenate(([0], delta_W_sem))


        plot_line(ax, poisson_rates, np.zeros(poisson_rates.shape), line_style='--', line_width=1.5, line_colour='grey')
        line_metadata = self.elem_details.get('delta_W')
        line = plot_line(ax, poisson_rates, delta_W, yerr=delta_W_sem, line_width=2, **line_metadata.to_kwargs())
        add_line_aligned_label(ax, 0, "LTP\u2191", x_offset=-5, y_offset=0.00)
        add_line_aligned_label(ax, 0, "LTD\u2193", x_offset=-5, y_offset=0.02, above=False,)

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_burst_probability(self, ax, ax_name='burst_probability'):
        poisson_rates = self.results.get_data('poisson_rate')
        burst_probability, burst_probability_sem = self.results.get_data('burst_probability')

        poisson_rates = np.concatenate(([0], poisson_rates))
        burst_probability = np.concatenate(([0], burst_probability))
        burst_probability_sem = np.concatenate(([0], burst_probability_sem))

        bp_line_metadata = self.elem_details.get('burst_probability')
        baseline_bp_line_metadata = self.elem_details.get('baseline_burst_probability')

        baseline_bp = plot_line(ax, poisson_rates, 50.0 * np.ones(poisson_rates.shape), line_width=1.5, **baseline_bp_line_metadata.to_kwargs())
        bp_line = plot_line(ax, poisson_rates, 100.0 * burst_probability, yerr=100.0 * burst_probability_sem, line_width=2, **bp_line_metadata.to_kwargs())

        ax.legend(
            handles=[baseline_bp],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.08),
            labelspacing=0.0,
            fontsize=12,
        )

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_event_burst_rates(self, ax, ax_name='event_burst_rates'):
        poisson_rates = self.results.get_data('poisson_rate')
        event_rate, event_rate_sem = self.results.get_data('event_rate')
        burst_rate, burst_rate_sem = self.results.get_data('burst_rate')
        single_spike_rate, single_spike_rate_sem = self.results.get_data('single_spike_rate')

        event_line_metadata = self.elem_details.get('event_rate')
        burst_line_metadata = self.elem_details.get('burst_rate')
        baseline_bp_line_metadata = self.elem_details.get('baseline_burst_probability')

        poisson_rates = np.concatenate(([0], poisson_rates))
        event_rate = np.concatenate(([0], event_rate))
        event_rate_sem = np.concatenate(([0], event_rate_sem))
        burst_rate = np.concatenate(([0], burst_rate))
        burst_rate_sem = np.concatenate(([0], burst_rate_sem))
        single_spike_rate = np.concatenate(([0], single_spike_rate))
        single_spike_rate_sem = np.concatenate(([0], single_spike_rate_sem))

        event_rate_line = plot_line(ax, poisson_rates, event_rate, yerr=event_rate_sem, line_width=2, **event_line_metadata.to_kwargs())
        half_event_line_kwargs = event_line_metadata.to_kwargs()
        half_event_line_kwargs["line_colour"] = baseline_bp_line_metadata.line_colour
        # event_rate_line2 = plot_line(ax, poisson_rates, event_rate*0.5, line_width=2, line_style='--', **half_event_line_kwargs)
        # single_spike_line = plot_line(
        #     ax,
        #     poisson_rates,
        #     single_spike_rate,
        #     yerr=single_spike_rate_sem,
        #     line_width=1.8,
        #     line_style=":",
        #     line_colour="dimgray",
        #     display_name="single-spike",
        # )
        burst_rate_line = plot_line(ax, poisson_rates, burst_rate, yerr=burst_rate_sem, line_width=2, **burst_line_metadata.to_kwargs())

        ax.legend(
            handles=[event_rate_line, burst_rate_line],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.18),
            labelspacing=0.0,
            fontsize=12,
        )

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "delta_W": {"fn": lambda p, ax: p.plot_delta_W(ax), "figsize": (4, 3)},
        "burst_probability": {"fn": lambda p, ax: p.plot_burst_probability(ax), "figsize": (4, 3)},
        "event_burst_rates": {"fn": lambda p, ax: p.plot_event_burst_rates(ax), "figsize": (4, 3)},
    }

    run_plots(PlasticityRulePlotter, PLOT_REGISTRY)
