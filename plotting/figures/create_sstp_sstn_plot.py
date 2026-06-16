from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.sstp_sstn import SSTpSSTnPlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 6.0
aspect_ratio = 1.2

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig)

root = ContainerNode(
    name="sstp_sstn_results",
    layout="column",
    spacings=[0.06],
    weights=[1.0, 1.0],
    children=[
        ContainerNode(
            layout="row",
            spacings=[0.02],
            weights=[1.0, 1.0],
            children=[
                PanelNode("data_cue_bars", inner_pad=(0.9, 0.7, 0.2, 0.1)),
                PanelNode("data_reward_delta_bars", inner_pad=(0.9, 0.7, 0.2, 0.1)),
            ],
        ),
        ContainerNode(
            layout="row",
            spacings=[0.02],
            weights=[1.0, 1.0],
            children=[
                PanelNode("model_cue_bars", inner_pad=(0.9, 0.7, 0.2, 0.1)),
                PanelNode("model_error_delta_bars", inner_pad=(0.9, 0.7, 0.2, 0.1)),
            ],
        ),
    ],
)

fig_manager.resolve_layout(root)

ax_data_cue = fig_manager.create_axes("data_cue_bars")
ax_data_reward = fig_manager.create_axes("data_reward_delta_bars")
ax_model_cue = fig_manager.create_axes("model_cue_bars")
ax_model_error = fig_manager.create_axes("model_error_delta_bars")

plotter = SSTpSSTnPlotter()
plotter.plot_data_cue_bars(ax_data_cue, show_legend=True)
plotter.plot_data_reward_delta_bars(ax_data_reward)
plotter.plot_model_cue_bars(ax_model_cue)
plotter.plot_model_error_delta_bars(ax_model_error)

fig_manager.finalise_figure("sstp_sstn.pdf", draw_debug=False, show=False)
