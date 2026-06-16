from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.dalean_bottleneck import DaleanBottleneckPlotter
from plotting.utils import init_global_matplotlib_constants


init_global_matplotlib_constants()

plot_width = 12.0
aspect_ratio = 0.55

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig, margin_inch=(0.45, 0.25, 0.20, 0.35))

panel_pad = (0.6, 0.82, 0.12, 0.10)
heatmap_panel_pad = (0.6, 0.82, 0.90, 0.10)

root = ContainerNode(
    name="dalean_bottleneck_sm",
    layout="column",
    spacings=[0.04],
    weights=[1.0, 1.0],
    children=[
        ContainerNode(
            name="equal_angle_row",
            layout="row",
            spacings=[0.04, 0.04],
            weights=[1.0, 1.0, 1.0],
            children=[
                PanelNode("equal_angle_fa_by_sst", inner_pad=panel_pad),
                PanelNode("equal_angle_bp_by_sst", inner_pad=panel_pad),
                PanelNode("equal_angle_wy_by_sst", inner_pad=panel_pad),
            ],
        ),
        ContainerNode(
            name="reduced_angle_heatmap_row",
            layout="row",
            spacings=[0.04, 0.04],
            weights=[1.0, 1.0, 1.0],
            children=[
                PanelNode("reduced_angle_fa_heatmap", inner_pad=heatmap_panel_pad),
                PanelNode("reduced_angle_bp_heatmap", inner_pad=heatmap_panel_pad),
                PanelNode("reduced_angle_wy_heatmap", inner_pad=heatmap_panel_pad),
            ],
        ),
    ],
)

fig_manager.resolve_layout(root)

fig_manager.add_label("equal_angle_row", "a")
fig_manager.add_label("reduced_angle_heatmap_row", "b")

axes = {
    name: fig_manager.create_axes(name)
    for name in [
        "equal_angle_fa_by_sst",
        "equal_angle_bp_by_sst",
        "equal_angle_wy_by_sst",
        "reduced_angle_fa_heatmap",
        "reduced_angle_bp_heatmap",
        "reduced_angle_wy_heatmap",
    ]
}

plotter = DaleanBottleneckPlotter()

plotter.plot_equal_angle_fa_by_sst(axes["equal_angle_fa_by_sst"])
plotter.plot_equal_angle_bp_by_sst(axes["equal_angle_bp_by_sst"])
plotter.plot_equal_angle_wy_by_sst(axes["equal_angle_wy_by_sst"])

plotter.plot_reduced_angle_fa_by_reduced_layer(axes["reduced_angle_fa_heatmap"])
plotter.plot_reduced_angle_bp_by_reduced_layer(axes["reduced_angle_bp_heatmap"])
plotter.plot_reduced_angle_wy_by_reduced_layer(axes["reduced_angle_wy_heatmap"])

for name in [
    "equal_angle_bp_by_sst",
    "equal_angle_wy_by_sst",
]:
    legend = axes[name].get_legend()
    if legend is not None:
        legend.remove()

fig_manager.finalise_figure("dalean_bottleneck_sm.pdf", draw_debug=False, show=False)
