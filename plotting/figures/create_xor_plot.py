from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.xor import XORPlotter, PhasePlotter

from plotting.utils import init_global_matplotlib_constants, add_vertical_span_across_axes, \
    add_vertical_line_across_axes

init_global_matplotlib_constants()

plot_width = 14
aspect_ratio = 0.45

fig = plt.figure(figsize=(plot_width, aspect_ratio * plot_width), constrained_layout=False)

fig_manager = FigureManager(fig)

root = ContainerNode(layout="row", spacings=[0.03], weights=[0.6, 1.3], children=[
    PanelNode("burstccn_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),

    # RIGHT COLUMN: phase panels on top, then result columns ordered by phase first
    ContainerNode(layout="column", spacings=[0.05], weights=[0.25, 1.0], children=[
        # ContainerNode(name="xor_phases", layout="row", spacings=[0.02], weights=[1, 1], children=[
        #     ContainerNode(layout="row", padding=(0.15, 0.0, 0.0, 0.0), spacings=[0.015], weights=[1, 1], children=[
        #         ContainerNode(layout="column", spacings=[0.015], weights=[1, 1], children=[
        #             PanelNode("xor_phases_burstprop_two_phase_teaching_signal", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #             PanelNode("xor_phases_burstprop_two_phase_plasticity", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #         ]),
        #         ContainerNode(layout="column", spacings=[0.015], weights=[1, 1], children=[
        #             PanelNode("xor_phases_burstccn_two_phase_teaching_signal", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #             PanelNode("xor_phases_burstccn_two_phase_plasticity", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #         ]),
        #     ]),
        #     ContainerNode(layout="row", padding=(0.15, 0.0, 0.0, 0.0), spacings=[0.015], weights=[1, 1], children=[
        #         ContainerNode(layout="column", spacings=[0.015], weights=[1, 1], children=[
        #             PanelNode("xor_phases_burstprop_one_phase_teaching_signal", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #             PanelNode("xor_phases_burstprop_one_phase_plasticity", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #         ]),
        #         ContainerNode(layout="column", spacings=[0.015], weights=[1, 1], children=[
        #             PanelNode("xor_phases_burstccn_one_phase_teaching_signal", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #             PanelNode("xor_phases_burstccn_one_phase_plasticity", inner_pad=(0.1, 0.0, 0.2, 0.0)),
        #         ]),
        #     ]),
        # ]),
        ContainerNode(name="xor_phases", layout="row", spacings=[0.02], weights=[1, 1], children=[
            ContainerNode(layout="column", padding=(0.0, 0.0, 0.0, 0.0), spacings=[0.055], weights=[1, 1], children=[
                PanelNode("xor_phases_two_phase_teaching_signal", inner_pad=(3.0, 0.0, 0.2, 0.0)),
                PanelNode("xor_phases_two_phase_plasticity", inner_pad=(3.0, 0.0, 0.2, 0.0)),
            ]),
            ContainerNode(layout="column", padding=(0.0, 0.0, 0.0, 0.0), spacings=[0.055], weights=[1, 1], children=[
                PanelNode("xor_phases_one_phase_teaching_signal", inner_pad=(1.5, 0.0, 1.0, 0.0)),
                PanelNode("xor_phases_one_phase_plasticity", inner_pad=(1.5, 0.0, 1.0, 0.0)),
            ]),
        ]),
        ContainerNode(layout="row", spacings=[0.02], weights=[1, 1], children=[
            ContainerNode(name="two_phase_results", layout="row", padding=(0.15, 0.0, 0.0, 0.0), spacings=[0.015], weights=[1, 1], children=[
                ContainerNode(name="burstccn_results", layout="column", spacings=[0.001, 0.001], weights=[1.1, 1, 1], children=[
                    PanelNode("burstccn_two_phase_event_rate", inner_pad=(0.1, 0.1, 0.2, 0.25)),
                    PanelNode("burstccn_two_phase_burst_probability", inner_pad=(0.1, 0.1, 0.2, 0.05)),
                    PanelNode("burstccn_two_phase_weight_change", inner_pad=(0.1, 0.3, 0.2, 0.05)),
                ]),
                ContainerNode(layout="column", spacings=[0.001, 0.001], weights=[1.1, 1, 1], children=[
                    PanelNode("burstprop_two_phase_event_rate", inner_pad=(0.1, 0.1, 0.2, 0.25)),
                    PanelNode("burstprop_two_phase_burst_probability", inner_pad=(0.1, 0.1, 0.2, 0.05)),
                    PanelNode("burstprop_two_phase_weight_change", inner_pad=(0.1, 0.3, 0.2, 0.05)),
                ]),
            ]),
            ContainerNode(name="one_phase_results", layout="row", padding=(0.15, 0.0, 0.0, 0.0), spacings=[0.015], weights=[1, 1], children=[
                ContainerNode(layout="column", spacings=[0.001, 0.001], weights=[1.1, 1, 1], children=[
                    PanelNode("burstccn_one_phase_event_rate", inner_pad=(0.1, 0.1, 0.2, 0.25)),
                    PanelNode("burstccn_one_phase_burst_probability", inner_pad=(0.1, 0.1, 0.2, 0.05)),
                    PanelNode("burstccn_one_phase_weight_change", inner_pad=(0.1, 0.3, 0.2, 0.05)),
                ]),
                ContainerNode(layout="column", spacings=[0.001, 0.001], weights=[1.1, 1, 1], children=[
                    PanelNode("burstprop_one_phase_event_rate", inner_pad=(0.1, 0.1, 0.2, 0.25)),
                    PanelNode("burstprop_one_phase_burst_probability", inner_pad=(0.1, 0.1, 0.2, 0.05)),
                    PanelNode("burstprop_one_phase_weight_change", inner_pad=(0.1, 0.3, 0.2, 0.05)),
                ]),
            ]),
        ]),
    ]),
])

fig_manager.resolve_layout(root)

fig_manager.add_label("burstccn_schematic", "a")
fig_manager.add_label("xor_phases", "b")
fig_manager.add_label("two_phase_results", "c")
fig_manager.add_label("one_phase_results", "d")


fig_manager.insert_pdf("burstccn_schematic", "xor/xor_burstccn_schematic4.pdf")
# fig_manager.insert_pdf("xor_phases", "xor/xor_phases2.pdf")

plotter = XORPlotter()
phase_plotter = PhasePlotter()
n_phase_examples = 2

phase_specs = [
    ("xor_phases_two_phase_teaching_signal", "two_phase", "teaching_signal", True, True, "two-phase learning"),
    ("xor_phases_two_phase_plasticity", "two_phase", "plasticity", True, True, None),
    ("xor_phases_one_phase_teaching_signal", "one_phase", "teaching_signal", False, True, "single-phase learning"),
    ("xor_phases_one_phase_plasticity", "one_phase", "plasticity", False, True, None),
]

for panel_name, phase_mode, signal_type, show_y_label, show_y_ticklabels, title in phase_specs:
    ax = fig_manager.create_axes(panel_name)
    phase_plotter.plot_phase_signal(
        ax,
        phase_mode=phase_mode,
        signal_type=signal_type,
        show_y_label=show_y_label,
        show_y_ticklabels=show_y_ticklabels,
        n_examples=n_phase_examples,
    )
    if title is not None:
        ax.set_title(title, fontweight="bold", fontfamily="Consolas", color="black", fontsize=15, pad=6)

for phase_mode in ["two_phase", "one_phase"]:
    ax_top = fig_manager.panels[f"xor_phases_{phase_mode}_teaching_signal"]["ax"]
    ax_bottom = fig_manager.panels[f"xor_phases_{phase_mode}_plasticity"]["ax"]
    for ex in range(n_phase_examples):
        add_vertical_span_across_axes(
            fig=fig_manager.fig,
            x_low=0.0 + ex * 1.0,
            x_high=1.0 + ex * 1.0,
            ax_low=ax_bottom,
            ax_high=ax_top,
            colour='#f0f0f0' if ex % 2 == 0 else '#d9d9d9',
            zorder=-2,
        )

# Maps each row index to the corresponding plotting function
row_plotters = [
    ("event_rate", plotter.plot_event_rate),
    ("burst_probability", plotter.plot_burst_probability),
    ("weight_change", plotter.plot_weight_changes),
]

# Phase modes and model types
col_order = [
    ("burstccn", "two_phase"),
    ("burstprop", "two_phase"),
    ("burstccn", "one_phase"),
    ("burstprop", "one_phase"),
]

axes = dict()
sharey_groups = {
    "two_phase": {
        "event_rate": "burstccn_two_phase_event_rate",
        "burst_probability": "burstccn_two_phase_burst_probability",
        "weight_change": "burstccn_two_phase_weight_change",
    },
    "one_phase": {
        "event_rate": "burstccn_one_phase_event_rate",
        "burst_probability": "burstccn_one_phase_burst_probability",
        "weight_change": "burstccn_one_phase_weight_change",
    },
}

for model_type, phase_mode in col_order:
    for row_name, plot_func in row_plotters:
        panel_name = f"{model_type}_{phase_mode}_{row_name}"
        sharey_name = sharey_groups[phase_mode][row_name]
        sharey_ax = axes.get(sharey_name, None)

        ax = fig_manager.create_axes(panel_name, sharey=sharey_ax)
        axes[panel_name] = ax
        show_y = (model_type == "burstccn")
        plot_func(ax, model_type=model_type, phase_mode=phase_mode, show_y_label=show_y)

for phase_mode in ["two_phase", "one_phase"]:
    for row_name, _ in row_plotters:
        ax = axes[f"burstprop_{phase_mode}_{row_name}"]
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")

model_titles = {
    "burstccn": "BurstCCN",
    "burstprop": "Burstprop",
}
for model_type in ["burstccn", "burstprop"]:
    axes[f"{model_type}_two_phase_event_rate"].set_title(
        model_titles[model_type],
        fontweight="bold",
        fontfamily="Consolas",
        color="black",
        fontsize=15,
        pad=6,
    )
    axes[f"{model_type}_one_phase_event_rate"].set_title(
        model_titles[model_type],
        fontweight="bold",
        fontfamily="Consolas",
        color="black",
        fontsize=15,
        pad=6,
    )


for model_type, phase_mode in col_order:
    ax_low = axes[f"{model_type}_{phase_mode}_weight_change"]
    ax_high = axes[f"{model_type}_{phase_mode}_event_rate"]
    for ex in range(4):
        add_vertical_span_across_axes(
            fig=fig_manager.fig,
            x_low=0.0 + ex * 8.0,
            x_high=8.0 + ex * 8.0,
            ax_low=ax_low,
            ax_high=ax_high,
            colour='#f0f0f0' if ex % 2 == 0 else '#d9d9d9',
            zorder=-2
        )
        # if phase_mode == 'two_phase':
        #     add_vertical_span_across_axes(
        #         fig=fig_manager.fig,
        #         x_low=7.2 + ex * 8.0 - 0.4,
        #         x_high=8.0 + ex * 8.0,
        #         ax_low=ax_low,
        #         ax_high=ax_high,
        #         colour='#eec2d7' #pink
        #     )

        # add_vertical_line_across_axes(
        #     fig=fig_manager.fig,
        #     x=0.0 + ex * 8.0,
        #     ax_low=ax_low,
        #     ax_high=ax_high,
        #     colour='#f0f0f0' if ex % 2 == 0 else '#d9d9d9'
        # )
        # if phase_mode == 'two_phase':
        #     add_vertical_line_across_axes(
        #         fig=fig_manager.fig,
        #         x=7.2 + ex * 8.0 - 0.4,
        #         ax_low=ax_low,
        #         ax_high=ax_high,
        #         colour='#DE85B0', #pink
        #         zorder=-1
        #     )

fig_manager.finalise_figure("xor.pdf", draw_debug=False)
