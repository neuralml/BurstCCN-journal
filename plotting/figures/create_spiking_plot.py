import matplotlib.pyplot as plt

from plotting.utils import init_global_matplotlib_constants, add_vertical_span_across_axes
from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode

from plotting.plotters.plasticity_rule import PlasticityRulePlotter
from plotting.plotters.stationary_target_spiking import StationaryTargetSpikingPlotter

init_global_matplotlib_constants()

plot_width = 14.5
aspect_ratio = 0.7

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig, margin_inch=(0.45, 0.05, 0.5, 0.45))

# Define layout tree
root = ContainerNode(layout="row", spacings=[0.015] * 2, weights=[0.5, 0.55, 1.0], children=[
    # LEFT column: header + 3 stacked plots
    ContainerNode(name="left_column", layout="column", spacings=[0.03], weights=[1.0, 1.0], children=[
        PanelNode("plasticity_rule_schematic", inner_pad=(0, 0, 0, 0)),
        ContainerNode(name="plasticity_rule_results", layout="column", spacings=[0.01] * 2, weights=[1.0, 1.0, 1.0],
                      children=[
                          PanelNode("delta_W", inner_pad=(0.8, 0.2, 0.1, 0.1)),
                          PanelNode("burst_probability", inner_pad=(0.8, 0.2, 0.1, 0.1)),
                          PanelNode("event_burst_rates", inner_pad=(0.8, 0.7, 0.1, 0.1)),
                      ])
    ]),
    PanelNode("spiking_schematic", inner_pad=(0, 0, 0, 0)),
    # RIGHT column: 6 stacked panels
    ContainerNode(name="spiking_results", layout="column", spacings=[0.01] * 5, weights=[1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
                  children=[
                      PanelNode("output_event_burst_rates", inner_pad=(1.1, 0.2, 0.2, 0.1)),
                      PanelNode("output_burst_probabilities", inner_pad=(1.1, 0.2, 0.2, 0.1)),
                      PanelNode("spike_trains", inner_pad=(1.1, 0.2, 0.2, 0.1)),
                      PanelNode("input_currents", inner_pad=(1.1, 0.2, 0.2, 0.1)),
                      PanelNode("dendritic_potentials", inner_pad=(1.1, 0.2, 0.2, 0.1)),
                      PanelNode("input_weight", inner_pad=(1.1, 0.6, 0.2, 0.1)),
                  ])
])

fig_manager.resolve_layout(root)

fig_manager.insert_pdf("plasticity_rule_schematic", "spiking/plasticity_rule_low_high.pdf")
fig_manager.insert_pdf("spiking_schematic", "spiking/spiking_schematic_clean_labelled.pdf")

ax_delta_W = fig_manager.create_axes("delta_W")
ax_burst_probability = fig_manager.create_axes("burst_probability", sharex=ax_delta_W)
ax_event_burst_rates = fig_manager.create_axes("event_burst_rates", sharex=ax_delta_W)

ax_output_event_burst_rates = fig_manager.create_axes("output_event_burst_rates")
ax_output_burst_probabilities = fig_manager.create_axes("output_burst_probabilities", sharex=ax_output_event_burst_rates)
ax_spike_trains = fig_manager.create_axes("spike_trains", sharex=ax_output_event_burst_rates)
ax_input_currents = fig_manager.create_axes("input_currents", sharex=ax_output_event_burst_rates)
ax_dendritic_potentials = fig_manager.create_axes("dendritic_potentials", sharex=ax_output_event_burst_rates)
ax_input_weight = fig_manager.create_axes("input_weight", sharex=ax_output_event_burst_rates)

pr_plotter = PlasticityRulePlotter()

pr_plotter.plot_event_burst_rates(ax_event_burst_rates)
pr_plotter.plot_burst_probability(ax_burst_probability)
pr_plotter.plot_delta_W(ax_delta_W)

st_plotter = StationaryTargetSpikingPlotter()

st_plotter.plot_event_burst_rates(ax_output_event_burst_rates)
ax_output_burst_probability_twin = st_plotter.plot_output_twin_dendritic_potentials_burst_probability(ax_output_burst_probabilities)
st_plotter.plot_spike_trains(ax_spike_trains)
st_plotter.plot_input_currents(ax_input_currents)
st_plotter.plot_twin_dendritic_potentials_burst_probability(ax_dendritic_potentials)
st_plotter.plot_input_weight(ax_input_weight)

axes_to_strip = [
    ax_delta_W,
    ax_burst_probability,
    ax_output_event_burst_rates,
    ax_output_burst_probabilities,
    ax_output_burst_probability_twin,
    ax_spike_trains,
    ax_input_currents,
    ax_dendritic_potentials,
]

for ax in axes_to_strip:
    ax.tick_params(labelbottom=False)
    # ax.set_xlabel("")

st_plotter.draw_inset_connection(
    fig_manager.fig,
    ax_top=ax_output_burst_probability_twin,
    ax_bottom=ax_spike_trains,
)


fig_manager.add_label("plasticity_rule_schematic", "a")
fig_manager.add_label("plasticity_rule_results", "b")
fig_manager.add_label("spiking_schematic", "c")
fig_manager.add_label("spiking_results", "d")

add_vertical_span_across_axes(fig_manager.fig, x_low=20 - 1.5, x_high=20 + 1.5,
                              ax_low=ax_event_burst_rates, ax_high=ax_delta_W, colour='#e5ffe5')

add_vertical_span_across_axes(fig_manager.fig, x_low=60 - 1.5, x_high=60 + 1.5,
                              ax_low=ax_event_burst_rates, ax_high=ax_delta_W, colour='#dcd4ec')

add_vertical_span_across_axes(fig_manager.fig, x_low=20000, x_high=40000,
                              ax_low=ax_input_weight, ax_high=ax_output_event_burst_rates, colour='#f0edea')

add_vertical_span_across_axes(fig_manager.fig, x_low=60000, x_high=80000,
                              ax_low=ax_input_weight, ax_high=ax_output_event_burst_rates, colour='#f0edea')

size = 0.23


def add_overlay_near_ylabel(fig_manager, parent_name, ax, overlay_name, pdf_file, size, x_pad_inch=0.06):
    """
    Place a numbered overlay relative to the rendered y-label of an Axes.

    The figure still uses the existing panel-relative overlay API internally;
    this helper just derives the local coordinates from the label bbox.
    """
    fig = fig_manager.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    label_bbox = ax.yaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    fig_w_inch, fig_h_inch = fig.get_size_inches()
    overlay_w_fig = size / fig_w_inch
    overlay_h_fig = size / fig_h_inch

    # Place the number just to the left of the label and vertically centered on it.
    overlay_x_fig = label_bbox.x0 - overlay_w_fig - (x_pad_inch / fig_w_inch)
    overlay_y_fig = label_bbox.y0 + 0.5 * (label_bbox.height - overlay_h_fig)

    parent_bbox = fig_manager.panels[parent_name]["bbox"]
    overlay_x = (overlay_x_fig - parent_bbox.x0) / parent_bbox.width
    overlay_y = (overlay_y_fig - parent_bbox.y0) / parent_bbox.height

    fig_manager.add_relative_overlay_pdf(
        parent_name=parent_name,
        name=overlay_name,
        pdf_file=pdf_file,
        x=overlay_x,
        y=overlay_y,
        size=size,
    )

spiking_number_overlays = [
    # (0.49, 0.56),
    (0.84, 0.875),
    (0.465, 0.78),
    (0.415, 0.45),
    (0.565, 0.37),
    (0.59, 0.095),
]

for idx, (x, y) in enumerate(spiking_number_overlays, start=1):
    fig_manager.add_relative_overlay_pdf(
        parent_name="spiking_schematic",
        name=f"spiking_num_overlay_{idx}",
        pdf_file=f"spiking/num_{idx}.pdf",
        x=x,
        y=y,
        size=size,
    )

spiking_results_number_targets = [
    (ax_output_event_burst_rates, "spiking_results_num_overlay_1", "spiking/num_1.pdf"),
    (ax_output_burst_probabilities, "spiking_results_num_overlay_2", "spiking/num_2.pdf"),
    (ax_input_currents, "spiking_results_num_overlay_3", "spiking/num_3.pdf"),
    (ax_dendritic_potentials, "spiking_results_num_overlay_4", "spiking/num_4.pdf"),
    (ax_input_weight, "spiking_results_num_overlay_5", "spiking/num_5.pdf"),
]

for ax, overlay_name, pdf_file in spiking_results_number_targets:
    add_overlay_near_ylabel(
        fig_manager=fig_manager,
        parent_name="spiking_results",
        ax=ax,
        overlay_name=overlay_name,
        pdf_file=pdf_file,
        size=size,
    )

fig_manager.finalise_figure("spiking.pdf", draw_debug=False)
