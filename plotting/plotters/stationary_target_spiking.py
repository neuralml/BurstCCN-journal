import numpy as np
from matplotlib.lines import Line2D
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter

from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis, transform_rect_to_fig_coords, plot_line

from plotting.analysis.stationary_target_spiking import StationaryTargetSpikingResultsStore
from plotting.plot_specs.stationary_target_spiking import StationaryTargetSpikingAxDetailsStore, \
    StationaryTargetSpikingElemDetailsStore


class StationaryTargetSpikingPlotter:
    def __init__(self):
        self.results = StationaryTargetSpikingResultsStore()
        self.ax_details = StationaryTargetSpikingAxDetailsStore()
        self.elem_details = StationaryTargetSpikingElemDetailsStore()

    def get_time_indices(self):
        """
        Returns an array of time indices from a given start time key to the experiment end time.

        Parameters:
        - start_time_key (str): Key in results settings for the start time (e.g. 'pre_learning_start_times')

        Returns:
        - indices (np.ndarray): Array of indices from start to experiment end
        """
        results_settings = self.results.get_settings()
        dt = results_settings["dt"]
        start_time = 10.0
        end_time = 90.002

        start_index = int(start_time / dt)
        end_index = int(end_time / dt)
        time_indices = np.arange(start_index, end_index)
        # times = self.convert_time_indices_to_times(time_indices)

        return time_indices

    def plot_event_burst_rates(self, ax, ax_name='event_burst_rates'):
        results_settings = self.results.get_settings()

        dt = results_settings["dt"]
        # pre_learning_start_times = results_settings["pre_learning_start_times"]
        during_learning_start_times = results_settings["during_learning_start_times"]
        during_learning_end_times = results_settings["during_learning_end_times"]

        time_indices = self.get_time_indices()

        event_rate = self.results.get_data('output_event_rates')
        burst_rate = self.results.get_data('output_burst_rates')
        target_rate = self.results.get_data('target_rates')

        event_line_metadata = self.elem_details.get('output_event_rates')
        burst_line_metadata = self.elem_details.get('output_burst_rates')
        target_line_metadata = self.elem_details.get('target_rates')

        plot_line(ax, time_indices, event_rate[time_indices], **event_line_metadata.to_kwargs())
        plot_line(ax, time_indices, burst_rate[time_indices], **burst_line_metadata.to_kwargs())

        target_line = np.full_like(target_rate, np.nan)  # fill everything with NaN

        for start, end in zip(during_learning_start_times, during_learning_end_times):
            s_idx, e_idx = int(start / dt), int(end / dt)
            target_line[int(start / dt):int(end / dt)] = target_rate[s_idx:e_idx]

        plot_line(ax, time_indices, target_line[time_indices], **target_line_metadata.to_kwargs())

        ax.legend(loc='upper right')

        ax_metadata = self.ax_details.get(ax_name)
        # setup_axis(ax, y_label=ax_metadata['y_label'], y_ticks=ax_metadata['y_ticks'])
        setup_axis(ax, **ax_metadata.to_kwargs())
        self.add_event_target_difference_arrows(ax, event_rate, target_rate, dt)

    def add_event_target_difference_arrows(self, ax, event_rate, target_rate, dt):
        plot_start_time = 10.0
        arrow_times = (10.0, 50.0)

        for arrow_time in arrow_times:
            x_idx = int((plot_start_time + arrow_time) / dt)
            y_event = event_rate[x_idx]
            y_target = target_rate[x_idx]

            ax.annotate(
                "",
                xy=(x_idx, y_target),
                xytext=(x_idx, y_event),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "linewidth": 1.2,
                    "mutation_scale": 8,
                },
                zorder=5,
            )

    def plot_burst_probabilities(self, ax, ax_name='output_burst_probability'):
        time_indices = self.get_time_indices()

        burst_probability = self.results.get_data('output_burst_probs_indirect')

        bp_line_metadata = self.elem_details.get('output_burst_prob')
        baseline_bp_line_metadata = self.elem_details.get('output_burst_prob_baseline')

        plot_line(ax, time_indices, 100.0 * burst_probability[time_indices], **bp_line_metadata.to_kwargs())
        baseline_bp_line = plot_line(ax, time_indices, np.repeat(0.35 * 100, len(time_indices)), **baseline_bp_line_metadata.to_kwargs())

        # ax.legend(loc='upper right')
        ax.legend(
            handles=[baseline_bp_line],
            loc='upper right',
            fontsize=12,
        )

        ax_metadata = self.ax_details.get(ax_name)
        # setup_axis(ax, y_label=ax_metadata['y_label'], y_ticks=ax_metadata['y_ticks'], y_lims=ax_metadata['y_lims'])
        setup_axis(ax, **ax_metadata.to_kwargs())

    def get_expanded_view_ranges(self):
        """
        Returns shared configuration for timeline expansion mapping.
        """
        full_timeline_ranges = [
            (10000, 20000),
            (20000, 40000),
            (40000, 60000),
            (60000, 80000),
            (80000, 90000)
        ]
        # expanded_view_ranges = [
        #     (13500, 16500),
        #     (22500, 28500),
        #     (47000, 53000),
        #     (62500, 68500),
        #     (83500, 86500)
        # ]

        expanded_view_ranges = [(14250, 15750),
                                (24000, 27000),
                                (48750, 51250),
                                (64250, 66750),
                                (84500, 85500)]

        padding = 800
        return full_timeline_ranges, expanded_view_ranges, padding

    def map_time_indices_to_expanded_view(self, indices):
        """
        Rescales 1D indices from full timeline segments to corresponding expanded view segments.
        """
        full_timeline_ranges, expanded_view_ranges, padding = self.get_expanded_view_ranges()

        if len(full_timeline_ranges) != len(expanded_view_ranges):
            raise ValueError("full_timeline_ranges and expanded_view_ranges must be the same length")

        clipped_ranges = [(start + padding, end - padding) for start, end in full_timeline_ranges]

        original_in_range = []
        transformed_indices = []

        for (clip_start, clip_end), (expanded_start, expanded_end) in zip(clipped_ranges, expanded_view_ranges):
            in_range = indices[(clip_start < indices) & (indices < clip_end)]
            rescaled = np.floor(
                (in_range - clip_start) / (clip_end - clip_start) * (expanded_end - expanded_start) + expanded_start
            )
            original_in_range.append(in_range)
            transformed_indices.append(rescaled)

        return np.concatenate(original_in_range).astype(int), np.concatenate(transformed_indices).astype(int)

    def plot_spike_trains(self, ax, ax_name='output_spike_trains'):
        """
        Plots spikes and bursts in an expanded x-axis format that zooms into different time regions.
        """
        time_indices = self.get_time_indices()

        input_events = self.results.get_data('input_events')
        input_bursts = self.results.get_data('input_bursts')

        pre_idx, transformed_idx = self.map_time_indices_to_expanded_view(time_indices)

        event_metadata = self.elem_details.get('input_events')
        burst_metadata = self.elem_details.get('input_bursts')

        def plot_vlines_multineuron(data, y_base, colour):
            n_neurons = 5
            row_height = 0.8 / n_neurons  # divide original 0.8 space into 5

            for neuron_idx in range(n_neurons):
                y_center = y_base - 0.4 + (neuron_idx + 0.5) * row_height
                positions = np.array([
                    pre_idx[i] for i, val in enumerate(data)
                    if val[neuron_idx] == 1
                ])
                ax.vlines(positions, y_center - row_height / 2, y_center + row_height / 2,
                          color=colour, linewidth=0.6)

        # Plot all 5 neurons for events and bursts
        plot_vlines_multineuron(input_events[transformed_idx], 3.0, event_metadata.line_colour)
        plot_vlines_multineuron(input_bursts[transformed_idx], 2.0, burst_metadata.line_colour)

        full_timeline_ranges, _, padding = self.get_expanded_view_ranges()
        label_x = full_timeline_ranges[0][0] - padding
        ax.text(label_x, 3.0, 'events', color=event_metadata.line_colour, ha='right', va='center')
        ax.text(label_x, 2.0, 'bursts', color=burst_metadata.line_colour, ha='right', va='center')

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, remove_axes=True, **ax_metadata.to_kwargs())

    def draw_inset_connection(self, fig, ax_top, ax_bottom, edgecolor='dimgray'):
        """
        Draws a path-only visual connection between expanded view segments in ax_bottom
        and their original locations in ax_top.
        """

        full_timeline_ranges, expanded_view_ranges, padding = self.get_expanded_view_ranges()
        clipped_ranges = [(start + padding, end - padding) for start, end in full_timeline_ranges]

        for (clip_start, clip_end), (expanded_start, expanded_end) in zip(clipped_ranges, expanded_view_ranges):
            # Region on top axis (the full timeline), showing where the expanded view came from
            expanded_range_on_top_axis = np.array([(expanded_start, 0.0), (expanded_end, 0.0)])

            # Box around the expanded view (on the bottom axis), showing the spike train window
            clipped_range_on_bottom_axis = np.array([(clip_start - padding // 2, 1.5), (clip_end + padding // 2, 3.5)])

            top_pos, top_w, top_h = transform_rect_to_fig_coords(fig, ax_top, expanded_range_on_top_axis)
            bottom_pos, bottom_w, bottom_h = transform_rect_to_fig_coords(fig, ax_bottom, clipped_range_on_bottom_axis)

            top_rect = patches.Rectangle(top_pos, top_w, top_h, transform=fig.transFigure)
            bottom_rect = patches.Rectangle(bottom_pos, bottom_w, bottom_h, transform=fig.transFigure)

            top_corners = top_rect.get_bbox().corners()
            bottom_corners = bottom_rect.get_bbox().corners()

            path_data = [
                # (Path.MOVETO, top_corners[0]),
                # (Path.LINETO, top_corners[1]),
                (Path.MOVETO, top_corners[3]),
                (Path.LINETO, top_corners[2]),
                (Path.LINETO, bottom_corners[3]),
                (Path.LINETO, bottom_corners[2]),
                (Path.LINETO, bottom_corners[0]),
                (Path.LINETO, bottom_corners[1]),
                (Path.LINETO, top_corners[0]),
                (Path.LINETO, top_corners[1]),
            ]

            # Add vertical caps to top edge
            cap_height = 0.005
            path_data.extend([
                (Path.MOVETO, (top_corners[3][0], top_corners[3][1] - cap_height)),
                (Path.LINETO, (top_corners[3][0], top_corners[3][1] + cap_height)),
                (Path.MOVETO, (top_corners[1][0], top_corners[1][1] - cap_height)),
                (Path.LINETO, (top_corners[1][0], top_corners[1][1] + cap_height)),
            ])

            path = Path([pt for code, pt in path_data], [code for code, pt in path_data])
            path_patch = patches.PathPatch(
                path, facecolor='none', edgecolor=edgecolor,
                linestyle='-', linewidth=1.5, joinstyle='round',
                transform=fig.transFigure
            )

            fig.patches.append(path_patch)

    def plot_input_currents(self, ax, ax_name='input_currents'):
        time_indices = self.get_time_indices()

        input_Q_inputs = self.results.get_data('input_Q_inputs', smoothing_sigma=50)
        input_Y_inputs = self.results.get_data('input_Y_inputs', smoothing_sigma=50)

        Q_line_metadata = self.elem_details.get('Q_input_current')
        Y_line_metadata = self.elem_details.get('Y_input_current')

        plot_line(ax, time_indices, 1e9 * input_Q_inputs[time_indices], **Q_line_metadata.to_kwargs())
        plot_line(ax, time_indices, 1e9 * input_Y_inputs[time_indices], **Y_line_metadata.to_kwargs())

        # ax.axhline(xmin=0, xmax=1.0, y=0, c='black', linestyle='--')
        # ax.legend(loc='upper right')
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.0, 1.08),
            fontsize=12,
        )

        ax_metadata = self.ax_details.get(ax_name)

        ax.spines['bottom'].set_position('zero')
        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_dendritic_potentials(self, ax, ax_name='dendritic_potentials'):
        time_indices = self.get_time_indices()

        leak_reversal_potential = self.results.get_settings().get('leak_reversal_potential', -70e-3)

        dendritic_potentials = self.results.get_data('input_dendritic_potentials', smoothing_sigma=50)

        dendritic_potential_line_metadata = self.elem_details.get('dendritic_potentials')
        reversal_potential_line_metadata = self.elem_details.get('dendritic_reversal_potential')

        plot_line(ax, time_indices, 1e3 * dendritic_potentials[time_indices], **dendritic_potential_line_metadata.to_kwargs())

        ax.axhline(xmin=0, xmax=1.0, y=leak_reversal_potential * 1e3,
                   label=reversal_potential_line_metadata.display_name,
                   color=reversal_potential_line_metadata.line_colour,
                   linestyle=reversal_potential_line_metadata.line_style,
                   zorder=-1)

        ax.legend(loc='upper right')

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_twin_dendritic_potentials_burst_probability(
        self,
        ax,
        ax_name='dendritic_potentials',
        dendritic_data_id='input_dendritic_potentials',
        burst_probability_data_id='input_burst_probs_indirect',
        baseline_burst_probability=0.375,
        dendritic_y_lims=(-85.0, -35.0),
        dendritic_y_ticks=(-80.0, -60.0, -40.0),
        burst_probability_y_lims=(0.0, 125.0),
        burst_probability_y_ticks=(0.0, 25.0, 50.0, 75.0, 100.0),
    ):
        time_indices = self.get_time_indices()

        leak_reversal_potential = self.results.get_settings().get('leak_reversal_potential', -70e-3)

        dendritic_potentials = self.results.get_data(dendritic_data_id, smoothing_sigma=50)
        burst_probability = self.results.get_data(burst_probability_data_id)

        dendritic_potential_line_metadata = self.elem_details.get('dendritic_potentials')
        reversal_potential_line_metadata = self.elem_details.get('dendritic_reversal_potential')
        bp_line_metadata = self.elem_details.get('output_burst_prob')
        baseline_bp_line_metadata = self.elem_details.get('output_burst_prob_baseline')

        dendritic_line = plot_line(
            ax,
            time_indices,
            1e3 * dendritic_potentials[time_indices],
            **dendritic_potential_line_metadata.to_kwargs(),
        )

        reversal_line = ax.axhline(
            xmin=0,
            xmax=1.0,
            y=leak_reversal_potential * 1e3,
            label=reversal_potential_line_metadata.display_name,
            color=reversal_potential_line_metadata.line_colour,
            linestyle=reversal_potential_line_metadata.line_style,
            zorder=-1,
        )

        ax_bp = ax.twinx()
        ax_bp.patch.set_visible(False)
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax_bp.set_zorder(1)

        bp_line = plot_line(
            ax_bp,
            time_indices,
            100.0 * burst_probability[time_indices],
            **bp_line_metadata.to_kwargs(),
        )
        baseline_bp_line = plot_line(
            ax_bp,
            time_indices,
            np.repeat(baseline_burst_probability * 100, len(time_indices)),
            **baseline_bp_line_metadata.to_kwargs(),
        )

        dash_length = 3.7
        dash_gap = 1.6
        dash_pattern = (dash_length, dash_gap + dash_length + dash_gap)
        reversal_line.set_linestyle((0.0, dash_pattern))
        baseline_bp_line.set_linestyle((dash_length + dash_gap, dash_pattern))
        reversal_line.set_zorder(-1)
        baseline_bp_line.set_zorder(-1)

        reversal_legend_line = Line2D(
            [],
            [],
            color=reversal_potential_line_metadata.line_colour,
            linestyle="--",
            label=reversal_potential_line_metadata.display_name,
        )
        baseline_legend_line = Line2D(
            [],
            [],
            color=baseline_bp_line_metadata.line_colour,
            linestyle="--",
            label=baseline_bp_line_metadata.display_name,
        )

        reversal_legend = ax.legend(
            [reversal_legend_line],
            [reversal_legend_line.get_label()],
            loc='upper left',
            bbox_to_anchor=(0.0, 1.08),
            fontsize=10,
        )
        ax.add_artist(reversal_legend)
        ax.legend(
            [baseline_legend_line],
            [baseline_legend_line.get_label()],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.08),
            fontsize=10,
        )

        ax_metadata = self.ax_details.get(ax_name)
        ax_kwargs = ax_metadata.to_kwargs()
        ax_kwargs["y_lims"] = dendritic_y_lims
        ax_kwargs["y_ticks"] = dendritic_y_ticks
        setup_axis(ax, **ax_kwargs)

        bp_axis_metadata = self.ax_details.get('output_burst_probability')
        setup_axis(
            ax_bp,
            y_label=bp_axis_metadata.y_label,
            y_lims=burst_probability_y_lims,
            y_ticks=burst_probability_y_ticks,
            twin_ax=True,
            ax_colour=bp_line_metadata.line_colour,
        )

        return ax_bp

    def plot_output_twin_dendritic_potentials_burst_probability(self, ax, ax_name='dendritic_potentials'):
        return self.plot_twin_dendritic_potentials_burst_probability(
            ax,
            ax_name=ax_name,
            dendritic_data_id='output_dendritic_potentials',
            burst_probability_data_id='output_burst_probs_indirect',
            baseline_burst_probability=0.35,
            dendritic_y_lims=(-85.0, -25.0),
            dendritic_y_ticks=(-80.0, -60.0, -40.0),
            burst_probability_y_lims=(0.0, 140.0),
            burst_probability_y_ticks=(0.0, 25.0, 50.0, 75.0, 100.0),
        )

    def plot_input_weight(self, ax, ax_name='input_weight'):
        dt = self.results.get_settings()['dt']
        time_indices = self.get_time_indices()

        input_weights = self.results.get_data('input_soma_input_weights')
        line_metadata = self.elem_details.get('input_weight')

        plot_line(ax, time_indices, input_weights[time_indices], **line_metadata.to_kwargs())

        ax_metadata = self.ax_details.get(ax_name)

        setup_axis(ax, **ax_metadata.to_kwargs())

        def time_formatter(x, pos):
            return f"{(x - 10000) * dt:.1f}"

        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    def draw_tall_rect(self, fig, x_low, x_high, ax_low, ax_high, facecolor='#f0edea'):
        """
        Draws a vertical rectangle spanning the full height from ax_low to ax_high
        at a given x-range, by transforming the corners into figure coordinates.

        Parameters:
        - x_low, x_high: Bounds of the rectangle in data coordinates.
        - ax_low, ax_high: The bottom and top axes to span.
        - fig: The matplotlib Figure to draw the rectangle on.
        - facecolor: Fill color for the rectangle.
        """
        rect_top = np.array([(x_low, ax_high.get_ylim()[0]), (x_high, ax_high.get_ylim()[1])])
        rect_bottom = np.array([(x_low, ax_low.get_ylim()[0]), (x_high, ax_low.get_ylim()[1])])

        top_loc, top_w, top_h = transform_rect_to_fig_coords(fig, ax_high, rect_top)
        bot_loc, bot_w, bot_h = transform_rect_to_fig_coords(fig, ax_low, rect_bottom)

        new_rect_height = top_loc[1] - bot_loc[1] + top_h

        rect = patches.Rectangle(bot_loc, bot_w, new_rect_height, facecolor=facecolor,
                                 zorder=-1, transform=fig.transFigure)

        fig.patches.append(rect)


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "event_burst_rates": {
            "fn": lambda p, ax: p.plot_event_burst_rates(ax),
            "figsize": (6, 3),
        },
        "burst_probabilities": {
            "fn": lambda p, ax: p.plot_burst_probabilities(ax),
            "figsize": (6, 3),
        },
        "spike_trains": {
            "fn": lambda p, ax: p.plot_spike_trains(ax),
            "figsize": (7, 2.5),
        },
        "input_currents": {
            "fn": lambda p, ax: p.plot_input_currents(ax),
            "figsize": (6, 3),
        },
        "dendritic_potentials": {
            "fn": lambda p, ax: p.plot_dendritic_potentials(ax),
            "figsize": (6, 3),
        },
        "twin_dendritic_potentials_burst_probability": {
            "fn": lambda p, ax: p.plot_twin_dendritic_potentials_burst_probability(ax),
            "figsize": (6, 3),
        },
        "output_twin_dendritic_potentials_burst_probability": {
            "fn": lambda p, ax: p.plot_output_twin_dendritic_potentials_burst_probability(ax),
            "figsize": (6, 3),
        },
        "input_weight": {
            "fn": lambda p, ax: p.plot_input_weight(ax),
            "figsize": (6, 2.5),
        },
    }

    # run_plots(StationaryTargetSpikingPlotter, PLOT_REGISTRY)
    run_plots(StationaryTargetSpikingPlotter, PLOT_REGISTRY, 'event_burst_rates')
