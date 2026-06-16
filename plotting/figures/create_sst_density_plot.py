from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.dalean_bottleneck import DaleanBottleneckPlotter
from plotting.plotters.sst_density import SSTDensityPlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

# plot_width = 7.0
# aspect_ratio = 1.0
#
# fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
# fig_manager = FigureManager(fig)

# root = ContainerNode(
#     layout="row",
#     spacings=[0.03],
#     weights=[0.95, 1.05],
#     children=[
#         PanelNode("brain_area_density", inner_pad=(0.55, 0.9, 0.15, 0.08)),
#         ContainerNode(
#             name="dalean_bottleneck_results",
#             layout="column",
#             spacings=[0.05],
#             weights=[1.0, 1.0],
#             children=[
#                 PanelNode("equal_bottleneck", inner_pad=(0.7, 0.5, 0.15, 0.08)),
#                 PanelNode("reduced_bottleneck", inner_pad=(0.7, 0.9, 0.15, 0.08)),
#             ],
#         ),
#     ],
# )

plot_width = 4.0
aspect_ratio = 2.61825726141

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig)

root = ContainerNode(
    layout="column",
    spacings=[0.01, 0.02],
    weights=[3.25, 1.0, 1.0],
    children=[
        PanelNode("brain_area_density", inner_pad=(0.55, 0.9, 0.15, 0.00)),
        PanelNode("equal_bottleneck", inner_pad=(0.55, 0.5, 0.15, 0.08)),
        PanelNode("reduced_bottleneck", inner_pad=(0.55, 0.9, 0.15, 0.08)),
    ],
)

fig_manager.resolve_layout(root)

# fig_manager.add_label("brain_area_density", "a")
# fig_manager.add_label("dalean_bottleneck_results", "b")

ax_brain_area_density = fig_manager.create_axes("brain_area_density")
ax_equal_bottleneck = fig_manager.create_axes("equal_bottleneck")
ax_reduced_bottleneck = fig_manager.create_axes("reduced_bottleneck")

sst_density_plotter = SSTDensityPlotter()
dalean_bottleneck_plotter = DaleanBottleneckPlotter()

sst_density_plotter.plot_brain_area_density(ax_brain_area_density)
dalean_bottleneck_plotter.plot_equal_bottleneck(ax_equal_bottleneck)
dalean_bottleneck_plotter.plot_reduced_bottleneck(ax_reduced_bottleneck)

fig_manager.finalise_figure("sst_density_and_dalean_bottleneck.pdf", draw_debug=False, show=False)
