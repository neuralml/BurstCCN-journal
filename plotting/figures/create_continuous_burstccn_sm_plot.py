from matplotlib import pyplot as plt

from plotting.plotters.continuous_burstccn import ContinuousBurstCCNPlotter
from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 13.0
aspect_ratio = 0.4

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio))
fig_manager = FigureManager(fig)

panel_names = [f"neuron_{i}" for i in range(10)]

top_row = [
    PanelNode(name, inner_pad=(0.3, 0.65, 0.15, 0.1), remove_y_axis=i > 0)
    for i, name in enumerate(panel_names[:5])
]
bottom_row = [
    PanelNode(name, inner_pad=(0.3, 0.65, 0.15, 0.1), remove_y_axis=i > 0)
    for i, name in enumerate(panel_names[5:])
]

root = ContainerNode(
    layout="column",
    spacings=[0.04],
    children=[
        ContainerNode(layout="row", spacings=[0.015, 0.015, 0.015, 0.015], children=top_row),
        ContainerNode(layout="row", spacings=[0.015, 0.015, 0.015, 0.015], children=bottom_row),
    ],
)

fig_manager.resolve_layout(root)

plotter = ContinuousBurstCCNPlotter()

task = "catcam"
seed = 1
start_step = 1
end_step = 10500001
start = 400
end = 3400

axes = []
for neuron_id, panel_name in enumerate(panel_names):
    ax = fig_manager.create_axes(panel_name)
    axes.append(ax)

    plotter.plot_output_comparison(
        ax,
        task=task,
        start_step=start_step,
        end_step=end_step,
        neuron_id=neuron_id,
        start=start,
        end=end,
        seed=seed,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Neuron {neuron_id}", fontweight="bold", fontfamily="Consolas", color="black", fontsize=15, pad=6)

for ax in axes[1:]:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

first_legend = axes[0].get_legend()
if first_legend is not None:
    first_legend.set_bbox_to_anchor((1.0, 0.0))

fig_manager.finalise_figure("continuous_burstccn_sm.pdf", show=True, draw_debug=False)
