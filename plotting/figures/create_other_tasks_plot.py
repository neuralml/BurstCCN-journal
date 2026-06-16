from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.image_tasks import ImageTaskPlotter
from plotting.plotters.rl import RLPlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 13.0
aspect_ratio = 0.625

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), dpi=90)
fig_manager = FigureManager(fig)

root = ContainerNode(
    name="rows_cifar_imagenet_grids",
    layout="column",
    spacings=[0.04, 0.04],  # row spacings
    weights=[1.0, 1.0, 1.1],
    children=[
        ContainerNode(
            name="cifar10_2x2",
            layout="row",
            spacings=[0.02, 0.02, 0.02],
            weights=[1.0, 1.0, 1.0, 1.0],
            children=[
                ContainerNode(
                    name="cifar10_schematics",
                    layout="row",
                    spacings=[0.01],
                    weights=[1.0, 2.0],
                    children=[
                        PanelNode("config_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                        PanelNode("cifar10_task_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                    ],
                ),
                PanelNode(
                    "cifar10_top1_test_performance",
                    inner_pad=(0.9, 0.65, 0.05, 0.05),
                    remove_x_axis=False,
                ),
                PanelNode(
                    "cifar10_BP_align",
                    inner_pad=(0.7, 0.65, 0.05, 0.05),
                    remove_x_axis=False,
                ),
                PanelNode(
                    "cifar10_WY_align",
                    inner_pad=(0.7, 0.65, 0.05, 0.05),
                ),
            ],
        ),

        # ---- RIGHT: ImageNet in a 2x2 grid (top-left blank)
        ContainerNode(
            name="imagenet_2x2",
            layout="row",
            spacings=[0.02, 0.02, 0.02],
            weights=[1.0, 1.0, 1.0, 1.0],
            children=[
                PanelNode("imagenet_task_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                PanelNode(
                    "imagenet_top5_test_performance",
                    inner_pad=(0.9, 0.65, 0.05, 0.05),
                    remove_x_axis=False,
                ),
                PanelNode(
                    "imagenet_BP_align",
                    inner_pad=(0.7, 0.65, 0.05, 0.05),
                    remove_x_axis=False,
                ),
                PanelNode(
                    "imagenet_WY_align",
                    inner_pad=(0.7, 0.65, 0.05, 0.05),
                ),
            ],
        ),
        ContainerNode(
            name="rl_4col",
            layout="row",
            spacings=[0.02, 0.02, 0.02],
            weights=[1.0, 1.0, 1.16, 0.84],
            children=[
                PanelNode(
                    "rl_task_schematic",
                    inner_pad=(0.05, 0.5, 1.0, 0.3)
                ),
                PanelNode(
                    "rl_performance",
                    inner_pad=(0.9, 1.0, 0.05, 0.05),
                ),
                PanelNode(
                    "rl_decoding_rl_variables",
                    # inner_pad=(2.1, 1.0, 0.05, 0.05),
                    inner_pad=(0.7, 1.0, 1.45, 0.05),
                ),
                PanelNode(
                    "rl_decoding_task_variables",
                    inner_pad=(0.7, 1.0, 0.05, 0.05),
                ),
            ],
        ),
    ],
)

fig_manager.resolve_layout(root)

fig_manager.add_label("config_schematic", "a")
fig_manager.add_label("cifar10_task_schematic", "b")
fig_manager.add_label("cifar10_top1_test_performance", "c")
fig_manager.add_label("imagenet_task_schematic", "d")
fig_manager.add_label("imagenet_top5_test_performance", "e")
fig_manager.add_label("rl_task_schematic", "f")
fig_manager.add_label("rl_performance", "g")
fig_manager.add_label("rl_decoding_rl_variables", "h")

# =========================
# ROW 1: MNIST axes
# =========================

# =========================
# ROW 2: CIFAR10 + ImageNet axes
# =========================

# CIFAR10 metrics
ax_cifar_top1 = fig_manager.create_axes("cifar10_top1_test_performance")
ax_cifar_bp = fig_manager.create_axes("cifar10_BP_align", sharex=ax_cifar_top1)
ax_cifar_wy = fig_manager.create_axes("cifar10_WY_align", sharex=ax_cifar_top1)

# ImageNet metrics
ax_imagenet_top1 = fig_manager.create_axes("imagenet_top5_test_performance")
ax_imagenet_bp = fig_manager.create_axes("imagenet_BP_align", sharex=ax_imagenet_top1)
ax_imagenet_wy = fig_manager.create_axes("imagenet_WY_align", sharex=ax_imagenet_top1)

# RL metrics
ax_rl_q_values = fig_manager.create_axes("rl_task_schematic")
ax_rl_performance = fig_manager.create_axes("rl_performance")
ax_rl_decoding_rl_variables = fig_manager.create_axes("rl_decoding_rl_variables")
ax_rl_decoding_task_variables = fig_manager.create_axes("rl_decoding_task_variables")

# =========================
# Insert schematics (PDFs)
# =========================
# Update paths to wherever your assets live.
# fig_manager.insert_pdf("mnist_schematic",          "schematics/mnist_task_schematic.pdf")
# fig_manager.insert_pdf("config_schematic", "Y_learning/burstccn_config_schematic_3.pdf")
fig_manager.insert_pdf("config_schematic", "Y_learning/burstccn_config_schematic_3_vert.pdf", align_x="center")
fig_manager.insert_pdf("cifar10_task_schematic",   "image_tasks/cifar-10_grid.pdf")
fig_manager.insert_pdf("imagenet_task_schematic", "image_tasks/imagenet_grid.pdf")
# fig_manager.insert_pdf("rl_task_schematic", "RL/RL_model_schematic.pdf")
# fig_manager.insert_pdf("rl_task_schematic", "RL/frozen_lake_Q_value_grid.pdf")

image_task_plotter = ImageTaskPlotter()
rl_plotter = RLPlotter()

# # --- CIFAR10
cifar_task = "cifar10"
# cifar_task = "imagenet"
image_task_plotter.plot_top1_test_performance(
    ax_cifar_top1,
    task=cifar_task,
    show_legend=True,
    legend_titles={
        "model": "Model",
        "mode": "Feedback type",
    },
    legend_locations={
        "model": "upper left",
        "mode": "upper left",
    },
    legend_bbox_to_anchor={
        "model": (0.08, 0.98),
        "mode": (0.45, 0.98),
    },
)
image_task_plotter.plot_BP_align(ax_cifar_bp, task=cifar_task)
image_task_plotter.plot_WY_align(ax_cifar_wy, task=cifar_task)

# --- ImageNet
inet_task = "imagenet"
image_task_plotter.plot_top5_test_performance(
    ax_imagenet_top1, task=inet_task, ax_name="top5_test_performance_imagenet"
)
image_task_plotter.plot_BP_align(ax_imagenet_bp, task=inet_task, ax_name="BP_align_imagenet")
image_task_plotter.plot_WY_align(ax_imagenet_wy, task=inet_task, ax_name="WY_align_imagenet")

# ax_cifar_top1.set_title("CIFAR10")
# ax_imagenet_top1.set_title("ImageNet")


rl_plotter.plot_Q_value_visualisation(ax_rl_q_values)
ax_rl_q_values.set_title(
    "Frozen Lake",
    fontsize=15,
    fontfamily="Consolas",
    fontweight="bold",
    pad=3,
)
rl_plotter.plot_avg_test_score(ax_rl_performance, show_legend=False)
ax_rl_performance.set_xlim(0, 30000)

rl_plotter.plot_decoding_accuracy_states_bar(
    ax_rl_decoding_rl_variables,
    decoder_target_type_group="rl_variables",
    exclude_predictor_data_types=["delta_b"],
    group_width=0.78,
    legend_bbox_to_anchor=(1.55, 0.42),
)
rl_plotter.plot_decoding_accuracy_states_bar(
    ax_rl_decoding_task_variables,
    decoder_target_type_group="task_variables",
    show_legend=False,
    exclude_predictor_data_types=["delta_b"],
    group_width=0.78,
)


fig_manager.finalise_figure(output_filename='other_tasks.pdf', draw_debug=False)

