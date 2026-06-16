from matplotlib import cm, colors, patches, ticker
import matplotlib.lines as mlines
import numpy as np

from plotting.analysis.rl import RLOfflineResultsStore, RLWandbResultsStore
from plotting.analysis.rl_model_utils import move_start_location
from plotting.plot_specs.rl import RLAxDetailsStore, RLElemDetailsStore
from plotting.plotters.plotter_base import run_plots
from plotting.utils import plot_line, setup_axis


class RLPlotter:
    def __init__(self):
        self.wandb_results = RLWandbResultsStore()
        self.offline_results = RLOfflineResultsStore()

        self.elem_details = RLElemDetailsStore()
        self.ax_details = RLAxDetailsStore()

    def get_decoding_chance_level(self, target_type, model_type="burstccn_fa", model_id=202):
        if target_type == "action":
            action = self.offline_results.get_predictor_data_from_actor(
                "action",
                model_type=model_type,
                model_id=model_id,
            )
            action_classes = np.argmax(action, axis=1)
            _, counts = np.unique(action_classes, return_counts=True)
            return counts.max() / counts.sum()

        if target_type == "agent_location":
            states, _ = self.offline_results.load_generated_states_data()
            locations = np.array([
                self.offline_results.get_agent_location(state, one_hot=False)
                for state in states
            ])
            _, counts = np.unique(locations, axis=0, return_counts=True)
            return counts.max() / counts.sum()

        if target_type == "relative_hole_locations":
            states, _ = self.offline_results.load_generated_states_data()
            hole_location_patterns = np.array([
                self.offline_results.calculate_relative_hole_locations(state)
                for state in states
            ])
            _, counts = np.unique(hole_location_patterns, axis=0, return_counts=True)
            return counts.max() / counts.sum()

        if target_type == "is_safe_action":
            safe_action = self.offline_results.get_decoder_target_data(
                "is_safe_action",
                model_type=model_type,
                model_id=model_id,
            )
            _, counts = np.unique(safe_action, return_counts=True)
            return counts.max() / counts.sum()

        return 0.0

    def plot_avg_test_score(
        self,
        ax,
        group="performance",
        model_type=None,
        sigma=1.5,
        smoothing_alpha=0.1,
        show_legend=True,
    ):
        ax_name = "avg_test_score"
        group_params = self.wandb_results.get_group_params(group)
        model_types = group_params["model_types"] if model_type is None else [model_type]
        modes = group_params["modes"]

        for mode in modes:
            mode_kwargs = self.elem_details.get(mode).to_kwargs()
            mode_line_style = mode_kwargs.get("line_style", "-")

            for model_type in model_types:
                episodes, mean, sem = self.wandb_results.get_mean_test_score_by_episode(
                    group=group,
                    model_type=model_type,
                    mode=mode,
                    sigma=sigma,
                )
                model_kwargs = self.elem_details.get(model_type).to_kwargs()
                plot_line(
                    ax,
                    episodes,
                    mean,
                    sem,
                    line_colour=model_kwargs.get("line_colour"),
                    line_style=mode_line_style,
                    smoothing_alpha=smoothing_alpha,
                )

        ax_metadata = self.ax_details.get(ax_name)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}"))
        setup_axis(ax, **ax_metadata.to_kwargs())

        if show_legend:
            mode_handles = []
            for mode in modes:
                mode_kwargs = self.elem_details.get(mode).to_kwargs()
                mode_handles.append(mlines.Line2D(
                    [],
                    [],
                    color="black",
                    linestyle=mode_kwargs.get("line_style", "-"),
                    label=mode_kwargs.get("display_name", mode),
                ))

            if mode_handles:
                mode_legend = ax.legend(handles=mode_handles, loc=ax_metadata.legend_location or "lower right")
                ax.add_artist(mode_legend)

            model_type_handles = []
            for model_type in model_types:
                model_kwargs = self.elem_details.get(model_type).to_kwargs()
                model_type_handles.append(mlines.Line2D(
                    [],
                    [],
                    color=model_kwargs.get("line_colour", "black"),
                    linestyle="solid",
                    label=model_kwargs.get("display_name", model_type),
                ))

            if model_type_handles:
                ax.legend(handles=model_type_handles, loc="upper right")

    def plot_Q_value_visualisation(
        self,
        ax,
        model_type="tmp",
        model_id=601,
        state_index=10,
        show_Q_values=True,
        show_q_value_text=False,
        show_colorbar=True,
        print_q_values=False,
        q_gamma=5.0,
        q_value_lims=None,
        highlight_optimal=True,
        optimal_edgecolor="#8E3AAB",
        optimal_linewidth=1.5,
    ):
        states, hole_maps, descriptors = self.offline_results.get_generated_states_and_maps()
        state = states[state_index]
        hole_map = hole_maps[state_index]
        map_desc = descriptors[state_index]
        q_values = self.offline_results.get_all_model_Q_values(
            model_type,
            model_id,
            hole_map,
            map_size=hole_map.shape[0],
        )

        if print_q_values:
            self._print_q_values(q_values, hole_map)

        env = self.offline_results.create_environment(move_start_location(map_desc, state))
        rendered = env.render()
        env_img = rendered[0] if isinstance(rendered, list) else rendered

        rows, cols = hole_map.shape
        ax.imshow(env_img, extent=[0, cols, 0, rows], aspect="equal")

        if show_Q_values:
            cmap = colors.LinearSegmentedColormap.from_list("q_value_white_high", ["white", "#007A59"])
            min_q, max_q = q_value_lims if q_value_lims is not None else (0.0, 1.0)

            def scale_q(q):
                q_clipped = np.clip(q, min_q, max_q)
                q_norm = (q_clipped - min_q) / (max_q - min_q + 1e-9)
                return q_norm ** q_gamma

            action_offsets = [
                (-1, 0),  # left
                (0, -1),  # down
                (1, 0),   # right
                (0, 1),   # up
            ]

            for row in range(rows):
                for col in range(cols):
                    if hole_map[row, col] == 1 or (row, col) == (rows - 1, cols - 1):
                        continue

                    flipped_row = rows - row - 1
                    center_x = col + 0.5
                    center_y = flipped_row + 0.5
                    state_q_values = q_values[row, col, : q_values.shape[-1]]
                    optimal_q_value = np.nanmax(state_q_values)

                    for action_index, (dx, dy) in enumerate(action_offsets[:q_values.shape[-1]]):
                        q_value = q_values[row, col, action_index]
                        is_optimal = highlight_optimal and np.isclose(q_value, optimal_q_value)
                        self._draw_q_value_arrow(
                            ax,
                            center_x,
                            center_y,
                            dx,
                            dy,
                            facecolor=cmap(scale_q(q_value)),
                            edgecolor=optimal_edgecolor if is_optimal else "black",
                            linewidth=optimal_linewidth if is_optimal else 0.5,
                            zorder=6 if is_optimal else 5,
                        )
                        if show_q_value_text:
                            self._draw_q_value_text(
                                ax,
                                center_x,
                                center_y,
                                dx,
                                dy,
                                q_value,
                            )

            if show_colorbar:
                sm = cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=min_q, vmax=max_q))
                sm.set_array([])
                colorbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                # colorbar.set_label("Action value (Q-value)")
                colorbar.set_label("Action value")

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _print_q_values(q_values, hole_map):
        action_names = ["left", "down", "right", "up"]
        rows, cols = hole_map.shape

        for row in range(rows):
            for col in range(cols):
                if hole_map[row, col] == 1:
                    print(f"state ({row}, {col}): hole")
                    continue

                values = ", ".join(
                    f"{action_names[i]}={q_values[row, col, i]:.3g}"
                    for i in range(min(len(action_names), q_values.shape[-1]))
                )
                print(f"state ({row}, {col}): {values}")

    @staticmethod
    def _draw_q_value_arrow(ax, cx, cy, dx, dy, facecolor, edgecolor="black", linewidth=0.5, zorder=5):
        norm = np.hypot(dx, dy)
        if norm == 0:
            return

        dx /= norm
        dy /= norm

        shaft_length = 0.1
        head_length = 0.15
        total_length = shaft_length + head_length
        width = 0.16

        tip_x = cx + dx * 0.465
        tip_y = cy + dy * 0.465
        tail_x = tip_x - dx * total_length
        tail_y = tip_y - dy * total_length
        head_base_x = tip_x - dx * head_length
        head_base_y = tip_y - dy * head_length

        pdx = -dy
        pdy = dx
        rear_left = (tail_x + pdx * width / 2, tail_y + pdy * width / 2)
        rear_right = (tail_x - pdx * width / 2, tail_y - pdy * width / 2)
        front_left = (head_base_x + pdx * width / 2, head_base_y + pdy * width / 2)
        front_right = (head_base_x - pdx * width / 2, head_base_y - pdy * width / 2)

        ax.add_patch(patches.Polygon(
            [rear_left, rear_right, front_right, (tip_x, tip_y), front_left],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            transform=ax.transData,
            zorder=zorder,
        ))

    @staticmethod
    def _draw_q_value_text(ax, cx, cy, dx, dy, q_value):
        norm = np.hypot(dx, dy)
        if norm == 0:
            return

        dx /= norm
        dy /= norm

        text_x = cx + dx * 0.28
        text_y = cy + dy * 0.28
        ax.text(
            text_x,
            text_y,
            f"{q_value:.3g}",
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            zorder=6,
        )

    def plot_decoding_accuracy_states_bar(
        self,
        ax,
        model_type="tmp",
        decoder_target_type_group="task_variables",
        model_id=50,
        show_legend=True,
        exclude_predictor_data_types=None,
        group_width=0.72,
        legend_bbox_to_anchor=(1.55, 0.5),
    ):
        model_type = model_type or "burstccn_fa"
        ax_name = f"decoding_{decoder_target_type_group}"

        if decoder_target_type_group == "task_variables":
            # decoder_target_types = ["action", "agent_location", "relative_hole_locations", "is_safe_action"]
            decoder_target_types = ["action", "agent_location", "relative_hole_locations"]
        elif decoder_target_type_group == "rl_variables":
            decoder_target_types = ["Q_value_prediction_errors", "action_Q_value", "distance_to_goal"]
        else:
            raise ValueError(f"Invalid decoder_target_type_group: {decoder_target_type_group}")

        predictor_data_types = [
            "somatic_potentials",
            "event_rates",
            "burst_rates",
            "apical_potentials",
            "burst_probabilities",
            "delta_b",
        ]
        if exclude_predictor_data_types is not None:
            excluded = set(exclude_predictor_data_types)
            predictor_data_types = [
                predictor_type
                for predictor_type in predictor_data_types
                if predictor_type not in excluded
            ]

        all_means = np.zeros((len(decoder_target_types), len(predictor_data_types)))
        all_stderrs = np.zeros((len(decoder_target_types), len(predictor_data_types)))

        for pi, predictor_type in enumerate(predictor_data_types):
            layer_names = ["fc2", "fc3"]

            for ti, target_type in enumerate(decoder_target_types):
                score_mean, score_stderr = self.offline_results.calculate_decoding_error(
                    model_type=model_type,
                    model_id=model_id,
                    predictor_data_type=predictor_type,
                    decoder_target_type=target_type,
                    layer_names=layer_names,
                )
                all_means[ti, pi] = score_mean
                all_stderrs[ti, pi] = score_stderr

        bar_width = group_width / len(predictor_data_types)
        x = np.arange(len(decoder_target_types))

        for pi, predictor_type in enumerate(predictor_data_types):
            elem_metadata = self.elem_details.get(predictor_type)
            elem_kwargs = elem_metadata.to_kwargs()
            ax.bar(
                x + pi * bar_width - ((len(predictor_data_types) - 1) / 2) * bar_width,
                all_means[:, pi],
                width=bar_width,
                yerr=all_stderrs[:, pi],
                capsize=1.5,
                color=elem_kwargs.get("line_colour"),
                label=elem_kwargs.get("display_name") if show_legend else None,
            )

        if decoder_target_type_group == "task_variables":
            for ti, target_type in enumerate(decoder_target_types):
                chance_level = self.get_decoding_chance_level(
                    target_type,
                    model_type=model_type,
                    model_id=model_id,
                )
                ax.hlines(
                    chance_level,
                    x[ti] - group_width / 2,
                    x[ti] + group_width / 2,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="chance" if ti == 0 else None,
                    zorder=3,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([
            self.elem_details.get(target_type).display_name for target_type in decoder_target_types
        ], rotation=35, ha="center")

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())
        ax.tick_params(axis="x", pad=-3)

        if decoder_target_type_group == "task_variables":
            ax.legend(loc="upper right", fontsize=8)
        elif show_legend:
            ax.legend(
                loc="center right",
                # bbox_to_anchor=(-0.22, 0.5),
                bbox_to_anchor=legend_bbox_to_anchor,
                borderaxespad=0.0,
                fontsize=9,
                # make legend bar glyph vertical
                handlelength=0.6,
                handleheight=1.8,
                handletextpad=0.75,
            )


if __name__ == "__main__":
    model_type = "tmp"
    model_id = 20

    PLOT_REGISTRY = {
        "avg_test_score": {
            "fn": lambda p, ax, **kw: p.plot_avg_test_score(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"show_legend": True},
        },
        "Q_value_visualisation": {
            "fn": lambda p, ax, **kw: p.plot_Q_value_visualisation(ax, **kw),
            "figsize": (3, 3),
            "kwargs": {
                "state_index": 1#14, # 1, 10, 14
            },
        },
        "decoding_rl_variables": {
            "fn": lambda p, ax, **kw: p.plot_decoding_accuracy_states_bar(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {
                "model_type": model_type,
                "model_id": model_id,
                "decoder_target_type_group": "rl_variables",
                "show_legend": True,
            },
        },
        "decoding_task_variables": {
            "fn": lambda p, ax, **kw: p.plot_decoding_accuracy_states_bar(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {
                "model_type": model_type,
                "model_id": model_id,
                "decoder_target_type_group": "task_variables",
                "show_legend": False,
            },
        },
    }

    # run_plots(RLPlotter, PLOT_REGISTRY, plot_names="avg_test_score")
    run_plots(RLPlotter, PLOT_REGISTRY, plot_names="Q_value_visualisation")
    # run_plots(RLPlotter, PLOT_REGISTRY, plot_names="decoding_rl_variables")
    # run_plots(RLPlotter, PLOT_REGISTRY, plot_names="decoding_task_variables")
