from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.mnist_Y_learning import MNISTPlotter
from plotting.utils import init_global_matplotlib_constants


init_global_matplotlib_constants()

plot_width = 12.0
aspect_ratio = 0.25

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), dpi=90)
fig_manager = FigureManager(fig, margin_inch=(0.25, 0.05, 0.05, 0.35))

root = ContainerNode(
    name="feedback_learning_scatter_row",
    layout="row",
    spacings=[0.03, 0.03],
    children=[
        PanelNode("qy_fa_branches", inner_pad=(0.65, 0.65, 1.2, 0.15)),
        PanelNode("qy_fa_noise", inner_pad=(0.65, 0.65, 1.2, 0.15)),
        PanelNode("qy_fa_error_scale", inner_pad=(0.65, 0.65, 1.2, 0.15)),
    ],
)

fig_manager.resolve_layout(root)

fig_manager.add_label("qy_fa_branches", "a")
fig_manager.add_label("qy_fa_noise", "b")
fig_manager.add_label("qy_fa_error_scale", "c")

ax_branches = fig_manager.create_axes("qy_fa_branches")
ax_noise = fig_manager.create_axes("qy_fa_noise")
ax_error_scale = fig_manager.create_axes("qy_fa_error_scale")

mnist_plotter = MNISTPlotter()

mnist_plotter.plot_QY_vs_FA_align_scatter(ax_branches, dataset="branches")
mnist_plotter.plot_QY_vs_FA_align_scatter(ax_noise, dataset="noise")
mnist_plotter.plot_QY_vs_FA_align_scatter(ax_error_scale, dataset="error_scale")

for ax in [ax_branches, ax_noise, ax_error_scale]:
    for collection in ax.collections:
        collection.set_alpha(0.7)

legend_titles = {
    ax_branches: "Apical\nbranches",
    ax_noise: "Noise std",
    ax_error_scale: "Teacher\nstrength",
}

for ax, title in legend_titles.items():
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title=title,
        loc="center right",
        bbox_to_anchor=(1.35, 0.5),
        markerscale=2.0,
    )

fig_manager.finalise_figure(output_filename="feedback_learning_sm.pdf", draw_debug=False)
