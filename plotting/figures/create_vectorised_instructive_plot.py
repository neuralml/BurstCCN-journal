from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.vectorised_instructive import VectorisedInstructivePlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 6.0
aspect_ratio = 1.2

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig)

root = ContainerNode(
    name="vectorised_instructive_results",
    layout="column",
    spacings=[0.06],
    weights=[1.0, 1.0],
    children=[
        ContainerNode(
            layout="row",
            spacings=[0.02],
            weights=[1.0, 1.0],
            children=[
                PanelNode("data_decreasing_error", inner_pad=(0.9, 0.7, 0.2, 0.1)),
                PanelNode("data_increasing_error", inner_pad=(0.9, 0.7, 0.2, 0.1)),
            ],
        ),
        ContainerNode(
            layout="row",
            spacings=[0.02],
            weights=[1.0, 1.0],
            children=[
                PanelNode("model_positive_target", inner_pad=(0.9, 0.7, 0.2, 0.1)),
                PanelNode("model_negative_target", inner_pad=(0.9, 0.7, 0.2, 0.1)),
            ],
        ),
    ],
)

fig_manager.resolve_layout(root)
# fig_manager.add_label("vectorised_instructive_results", "a")

ax_data_dec = fig_manager.create_axes("data_decreasing_error")
ax_data_inc = fig_manager.create_axes("data_increasing_error", sharey=ax_data_dec)
ax_model_pos = fig_manager.create_axes("model_positive_target", sharey=ax_data_dec)
ax_model_neg = fig_manager.create_axes("model_negative_target", sharey=ax_data_dec)

plotter = VectorisedInstructivePlotter()
plotter.plot_data_condition(ax_data_dec, condition="decreasing_error", show_legend=True)
plotter.plot_data_condition(ax_data_inc, condition="increasing_error")
plotter.plot_model_condition(ax_model_pos, condition="positive_target")
plotter.plot_model_condition(ax_model_neg, condition="negative_target")

fig_manager.finalise_figure("vectorised_instructive.pdf", draw_debug=False, show=False)
