from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.mnist_Y_learning import MNISTPlotter, MNISTApicalActivityPlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 15.0
aspect_ratio = 0.45

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), dpi=90)
fig_manager = FigureManager(fig)

root = ContainerNode(
            name="top_qy_and_activity",
            layout="row",
            spacings=[0.03],  # 3 columns -> 2 spacings
            weights=[1.2, 1.0],
            children=[
                ContainerNode(
                    name="activity_and_y_metrics",
                    layout="column",
                    spacings=[0.06],  # activity row vs metrics row
                    weights=[1.0, 3.0],
                    children=[
                        ContainerNode(
                            name="y_metrics_row",
                            layout="row",
                            spacings=[0.02, 0.005],  # 2 children -> 1 spacing
                            weights=[1.7, 1.0, 1.0],
                            children=[
                                PanelNode("y_learning_schematic"),
                                PanelNode("y_metric_apical", inner_pad=(1.1, 0.45, 0.0, 0.0)),
                                PanelNode("y_metric_burst_prob", inner_pad=(1.1, 0.45, 0.0, 0.0)),
                                # PanelNode("y_metric_fa_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                            ],
                        ),
                        # Top: BEFORE/AFTER activity (scatter + Exc/Inh stacked)
                        ContainerNode(
                            name="activity_before_after_row",
                            layout="row",
                            spacings=[0.03],  # 2 columns -> 1 spacing
                            children=[
                                # BEFORE: scatter + Exc/Inh stacked
                                ContainerNode(
                                    name="before_activity",
                                    layout="column",
                                    spacings=[0.02],  # 2 panels -> 1 spacing
                                    weights=[1.25, 1.0],
                                    children=[
                                        PanelNode("before_scatter", inner_pad=(0.95, 0.5, 0.05, 0.3)),
                                        PanelNode("before_excinh",  inner_pad=(0.95, 0.7, 0.05, 0.05)),
                                    ],
                                ),
                                # AFTER: scatter + Exc/Inh stacked
                                ContainerNode(
                                    name="after_activity",
                                    layout="column",
                                    spacings=[0.02],
                                    weights=[1.25, 1.0],
                                    children=[
                                        PanelNode("after_scatter", inner_pad=(0.95, 0.5, 0.05, 0.3)),
                                        PanelNode("after_excinh",  inner_pad=(0.95, 0.7, 0.05, 0.05)),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                ContainerNode(
                    name="qy_panels_stacked",
                    layout="column",
                    spacings=[0.02, 0.02],  # 3 panels -> 2 spacings
                    children=[
                        ContainerNode(
                            name="qy_noise_container",
                            layout="row",
                            spacings=[0.00],
                            weights=[1, 1.5],
                            children=[
                                PanelNode("qy_noise_schematic"),
                                PanelNode("qy_noise", inner_pad=(0.5, 0.7, 1.5, 0.05)),
                            ],
                        ),
                        ContainerNode(
                            name="qy_branches_container",
                            layout="row",
                            spacings=[0.00],
                            weights=[1, 1.5],
                            children=[
                                PanelNode("qy_branches_schematic"),
                                PanelNode("qy_branches", inner_pad=(0.5, 0.7, 1.5, 0.05)),
                            ],
                        ),
                        ContainerNode(
                            name="qy_error_container",
                            layout="row",
                            spacings=[0.00],
                            weights=[1, 1.5],
                            children=[
                                PanelNode("qy_error_schematic"),
                                PanelNode("qy_error", inner_pad=(0.5, 0.7, 1.5, 0.05)),
                            ],
                        ),
                    ],
                ),

            ],
        )

fig_manager.resolve_layout(root)

fig_manager.add_label("y_learning_schematic", "a")
fig_manager.add_label("y_metric_apical", "b")
fig_manager.add_label("before_scatter", "c")
# fig_manager.add_label("after_scatter", "d")
fig_manager.add_label("before_excinh", "d")
fig_manager.add_label("qy_noise_container", "e")
fig_manager.add_label("qy_branches_container", "f")
fig_manager.add_label("qy_error_container", "g")

ax_qy_noise    = fig_manager.create_axes("qy_noise")
ax_qy_branches = fig_manager.create_axes("qy_branches", sharex=ax_qy_noise)
ax_qy_error    = fig_manager.create_axes("qy_error", sharex=ax_qy_noise)

# Row 2: activity + Y metrics4
ax_before_scatter = fig_manager.create_axes("before_scatter")
ax_before_excinh  = fig_manager.create_axes("before_excinh")
ax_after_scatter  = fig_manager.create_axes("after_scatter", sharey=ax_before_scatter)
ax_after_excinh   = fig_manager.create_axes("after_excinh", sharey=ax_before_excinh)
ax_y_metric_apical = fig_manager.create_axes("y_metric_apical")
# ax_y_metric_fa     = fig_manager.create_axes("y_metric_fa_angle")
ax_y_metric_burst_prob = fig_manager.create_axes("y_metric_burst_prob")

fig_manager.insert_pdf(
    "y_learning_schematic",
    "Y_learning/burstccn_config_schematic_1.pdf",
    align_y="center",
)
fig_manager.insert_pdf("qy_noise_schematic", "Y_learning/burstccn_feedback_noise.pdf")
fig_manager.insert_pdf("qy_branches_schematic", "Y_learning/burstccn_feedback_branches.pdf")
fig_manager.insert_pdf("qy_error_schematic", "Y_learning/burstccn_feedback_teacher_strength.pdf")

mnist_plotter = MNISTPlotter()
activity_plotter = MNISTApicalActivityPlotter()

mnist_plotter.plot_Y_learning_apical_magnitude(ax_y_metric_apical)
# mnist_plotter.plot_Y_learning_angle_fa(ax_y_metric_fa)
mnist_plotter.plot_Y_learning_burst_prob_magnitude(ax_y_metric_burst_prob)

activity_plotter.plot_apical_QY_scatter_before(ax_before_scatter)
activity_plotter.plot_apical_exc_inh_inputs_before(ax_before_excinh, show_legend=False)
activity_plotter.plot_apical_QY_scatter_after(ax_after_scatter, show_legend=False)
activity_plotter.plot_apical_exc_inh_inputs_after(ax_after_excinh, show_legend=True)

for ax in [ax_after_scatter, ax_after_excinh]:
    ax.tick_params(labelleft=False)
    ax.set_ylabel("")

mnist_plotter.plot_QY_align_across_noise(ax_qy_noise)
mnist_plotter.plot_QY_align_across_branches(ax_qy_branches)
mnist_plotter.plot_QY_align_across_error_scale(ax_qy_error)

fig_manager.finalise_figure(output_filename='feedback_learning.pdf')
