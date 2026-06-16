from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.mnist_Y_learning import MNISTPlotter, MNISTApicalActivityPlotter
from plotting.plotters.mnist_representation import MNISTRepresentationPlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

# plot_width = 13.0
# aspect_ratio = 0.9

plot_width = 15.0
aspect_ratio = 0.8

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), dpi=90)
fig_manager = FigureManager(fig)

root = ContainerNode(
    name="root",
    layout="column",
    spacings=[0.04, 0.04],  # 2 columns -> 1 spacing
    weights=[2.5, 1.0, 1.0],
    children=[
        ContainerNode(
            name="wy_learning",
            layout="row",
            spacings=[0.03],  # 2 columns -> 1 spacing
            weights=[0.45, 1.0],
            children=[
                ContainerNode(
                    name="schematic_container",
                    layout="column",
                    spacings=[0.04],
                    weights=[0.4, 1.0],
                    children=[
                        PanelNode("wy_learning_schematic"),
                        PanelNode("block_training_schematic"),
                    ]
                ),

                # Column 2: transposed 3x3 grid as 3 stacked rows
                ContainerNode(
                    name="wy_learning_grid_rows",
                    layout="column",
                    spacings=[0.03, 0.03],  # 3 rows -> 2 spacings
                    children=[
                        ContainerNode(
                            name="wy_learning_metrics",
                            layout="row",
                            spacings=[0.03, 0.03],  # 3 columns -> 2 spacings
                            children=[
                                PanelNode("wy_learning_test_error", inner_pad=(0.85, 0.5, 0.2, 0.05)),
                                PanelNode("wy_learning_qy_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                                PanelNode("wy_learning_fa_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                                # PanelNode("wy_learning_bp_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                            ],
                        ),
                        ContainerNode(
                            name="wy_learning_branches",
                            layout="row",
                            spacings=[0.03, 0.03],  # 3 columns -> 2 spacings
                            children=[
                                PanelNode("wy_learning_branches_test_error", inner_pad=(0.85, 0.5, 0.2, 0.05)),
                                PanelNode("wy_learning_branches_qy_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                                PanelNode("wy_learning_branches_fa_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                                # PanelNode("wy_learning_branches_bp_angle", inner_pad=(0.5, 0.5, 0.2, 0.05)),
                            ],
                        ),
                        ContainerNode(
                            name="without_teacher",
                            layout="row",
                            spacings=[0.03, 0.03],  # 3 columns -> 2 spacings
                            children=[
                                # PanelNode("without_teacher_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                                PanelNode("without_teacher_blocks_test_error", inner_pad=(0.85, 0.6, 0.2, 0.05)),
                                PanelNode("without_teacher_blocks_qy_angle", inner_pad=(0.5, 0.6, 0.2, 0.05)),
                                PanelNode("without_teacher_blocks_bp_angle", inner_pad=(0.5, 0.6, 0.2, 0.05)),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        ContainerNode(
            name="mnist_depth_plots_row",
            layout="row",
            spacings=[0.03, 0.03, 0.03],  # 3 panels -> 2 spacings
            weights=[1.0, 1.0, 1.0],
            children=[
                PanelNode(
                    "network_depth_test_errors",
                    inner_pad=(0.5, 0.5, 0.2, 0.05),
                    remove_x_axis=False,

                ),
                PanelNode(
                    "network_depth_FA_align",
                    inner_pad=(0.5, 0.5, 0.2, 0.05),
                ),
                PanelNode(
                    "network_depth_BP_align",
                    inner_pad=(0.5, 0.5, 0.2, 0.05),
                    remove_x_axis=False,
                ),
            ],
        ),
        ContainerNode(
            name="mnist_representation_row",
            layout="row",
            spacings=[0.03, 0.03, 0.03],  # 3 panels -> 2 spacings
            weights=[1.0, 1.0, 1.1, 1.2],
            children=[
                PanelNode("mnist_tsne_ann", inner_pad=(0.5, 0.2, 0.1, 0.05)),
                PanelNode("mnist_tsne_burstccn_online", inner_pad=(0.5, 0.2, 0.1, 0.05)),
                PanelNode("mnist_tsne_burst_prob", inner_pad=(0.5, 0.2, 0.7, 0.05)),
                PanelNode("mnist_mds", inner_pad=(0.5, 0.2, 1.55, 0.05)),
            ],
        ),
    ],
)

fig_manager.resolve_layout(root)

fig_manager.add_label("wy_learning_schematic", "a")
fig_manager.add_label("block_training_schematic", "b")
fig_manager.add_label("wy_learning_metrics", "c")
fig_manager.add_label("wy_learning_branches", "d")
fig_manager.add_label("without_teacher", "e")
fig_manager.add_label("network_depth_test_errors", "f")
fig_manager.add_label("mnist_tsne_ann", "g")
fig_manager.add_label("mnist_tsne_burst_prob", "h")
fig_manager.add_label("mnist_mds", "i")

# Row 3: Y-learning plots (not the schematic)
ax_wy_learning_err = fig_manager.create_axes("wy_learning_test_error")
ax_wy_learning_qy_angle = fig_manager.create_axes("wy_learning_qy_angle")
ax_wy_learning_fa_angle  = fig_manager.create_axes("wy_learning_fa_angle")
# ax_wy_learning_bp_angle = fig_manager.create_axes("wy_learning_bp_angle")
# ax_wy_learning_ablate = fig_manager.create_axes("wy_learning_ablate_bar")
ax_wy_learning_branches_test_error = fig_manager.create_axes("wy_learning_branches_test_error")
ax_wy_learning_branches_qy_angle = fig_manager.create_axes("wy_learning_branches_qy_angle")
ax_wy_learning_branches_fa_angle = fig_manager.create_axes("wy_learning_branches_fa_angle")
# ax_wy_learning_branches_bp_angle = fig_manager.create_axes("wy_learning_branches_bp_angle")

# Row 4: only one plot; the others are schematic/empty
# ax_without_teacher = fig_manager.create_axes("without_teacher_blocks")
ax_without_teacher_test_error = fig_manager.create_axes("without_teacher_blocks_test_error")
ax_without_teacher_qy_angle = fig_manager.create_axes("without_teacher_blocks_qy_angle")
ax_without_teacher_bp_angle = fig_manager.create_axes("without_teacher_blocks_bp_angle")

fig_manager.insert_pdf("wy_learning_schematic", "Y_learning/burstccn_config_schematic_2.pdf")
fig_manager.insert_pdf("block_training_schematic", "Y_learning/block_training_schematic_6.pdf")

ax_mnist_depth_test_errors = fig_manager.create_axes("network_depth_test_errors")
ax_mnist_depth_bp_align = fig_manager.create_axes("network_depth_BP_align", sharex=ax_mnist_depth_test_errors)
ax_mnist_depth_fa_align = fig_manager.create_axes("network_depth_FA_align", sharex=ax_mnist_depth_test_errors)

ax_tsne_ann = fig_manager.create_axes("mnist_tsne_ann")
ax_tsne_burstccn_online = fig_manager.create_axes("mnist_tsne_burstccn_online")
ax_tsne_burst_prob = fig_manager.create_axes("mnist_tsne_burst_prob")
ax_mnist_mds = fig_manager.create_axes("mnist_mds")

mnist_plotter = MNISTPlotter()
activity_plotter = MNISTApicalActivityPlotter()
repr_plotter = MNISTRepresentationPlotter()  # e.g. TSNE/MDS

mnist_plotter.plot_with_without_Y_learning_test_performance(ax_wy_learning_err)
mnist_plotter.plot_with_without_Y_learning_QY_align(ax_wy_learning_qy_angle, show_legend=False)
mnist_plotter.plot_with_without_Y_learning_FA_align(ax_wy_learning_fa_angle, show_legend=False)
# mnist_plotter.plot_with_without_Y_learning_BP_align(ax_wy_learning_bp_angle, show_legend=False)

ax_wy_learning_err.set_yticks([0, 25, 50, 75, 100])
ax_wy_learning_qy_angle.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
# ax_wy_learning_fa_angle.set_yticks([25, 50, 75, 100])
ax_wy_learning_fa_angle.set_yticks([0, 50, 100])

mnist_plotter.plot_across_apical_branches_test_error(ax_wy_learning_branches_test_error)
mnist_plotter.plot_across_apical_branches_QY_align(ax_wy_learning_branches_qy_angle)
mnist_plotter.plot_across_apical_branches_FA_align(ax_wy_learning_branches_fa_angle)
# mnist_plotter.plot_across_apical_branches_BP_align(ax_wy_learning_branches_bp_angle)

ax_wy_learning_branches_test_error.set_yticks([3.0, 3.5, 4.0])
ax_wy_learning_branches_qy_angle.set_yticks([0.2, 0.3, 0.4])
# ax_wy_learning_branches_fa_angle.set_yticks([25, 30, 35, 40, 45])
ax_wy_learning_branches_fa_angle.set_yticks([25, 35, 45])

mnist_plotter.plot_without_teacher_blocks_test_error(ax_without_teacher_test_error)
mnist_plotter.plot_without_teacher_blocks_QY_align(ax_without_teacher_qy_angle)
mnist_plotter.plot_without_teacher_blocks_FA_align(ax_without_teacher_bp_angle)

# ax_without_teacher_test_error.set_yticks([1.75, 2.0, 2.25, 2.5, 2.75])
ax_without_teacher_test_error.set_yticks([1.75, 2.25, 2.75])
# ax_without_teacher_qy_angle.set_yticks([0.175, 0.2, 0.225, 0.25])
ax_without_teacher_qy_angle.set_yticks([0.18, 0.22, 0.26])
ax_without_teacher_bp_angle.set_yticks([5, 15, 25])

mnist_plotter.plot_network_depth_test_errors(ax_mnist_depth_test_errors, show_legend=True)
mnist_plotter.plot_network_depth_FA_align(ax_mnist_depth_fa_align)
mnist_plotter.plot_network_depth_BP_align(ax_mnist_depth_bp_align)

ax_mnist_depth_fa_align.set_yticks([0, 10, 20, 30, 40])

repr_plotter.plot_TSNE(ax_tsne_ann, model_name="ann", show_legend=True)
repr_plotter.plot_TSNE(ax_tsne_burstccn_online, model_name="burstccn")
repr_plotter.plot_TSNE(ax_tsne_burst_prob, model_name="burstccn", colour_by='burst_prob')
# repr_plotter.plot_TSNE(ax_tsne_burstccn_hybrid, model_name="burstccn_Y_block_trained")
# repr_plotter.plot_TSNE(ax_tsne_burstccn_offline,  model_name="burstccn_QY_tied")

repr_plotter.plot_MDS(ax_mnist_mds)

fig_manager.finalise_figure(output_filename='mnist.pdf')
