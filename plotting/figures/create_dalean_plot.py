from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from string import ascii_lowercase

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.dalean_ablation import DaleanAblationPlotter
from plotting.plotters.dalean_bottleneck import DaleanBottleneckPlotter
from plotting.plotters.sst_density import SSTDensityPlotter
from plotting.plotters.sstp_sstn import SSTpSSTnPlotter
from plotting.plotters.vectorised_instructive import VectorisedInstructivePlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

available_fonts = {font.name for font in fm.fontManager.ttflist}
ref_fontfamily = "Open Sans" if "Open Sans" in available_fonts else "Liberation Sans"

plot_width = 14.0
aspect_ratio = 0.92

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig, margin_inch=(0.45, 0.2, 0.45, 0.6))

root = ContainerNode(
    layout="column",
    spacings=[0.06],
    weights=[1.975, 1.0],
    children=[
        ContainerNode(
            layout="row",
            spacings=[0.02],
            weights=[1.0, 1.75],
            children=[
                ContainerNode(
                    layout="column",
                    spacings=[0.05],
                    weights=[2.5, 1.0],
                    children=[
                        PanelNode("dalean_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                        ContainerNode(
                            name="dalean_ablation_results",
                            layout="row",
                            spacings=[0.02],
                            weights=[1.0, 1.0],
                            children=[
                                PanelNode("ablation_data", inner_pad=(1.20, 0.60, 0.18, 0.10)),
                                PanelNode("ablation_model_weight_change", inner_pad=(0.95, 0.60, 0.18, 0.10)),
                            ],
                        ),
                    ],
                ),
                ContainerNode(
                    name="right_column_results",
                    layout="column",
                    spacings=[0.06],
                    weights=[1.30, 1.0],
                    children=[
                        ContainerNode(
                            name="sst_density_results",
                            layout="row",
                            spacings=[0.02],
                            weights=[1.1, 2.0],
                            children=[
                                PanelNode("brain_area_density", inner_pad=(1.2, 0.75, 0.14, 0.04)),
                                ContainerNode(
                                    name="dalean_bottleneck_results",
                                    layout="column",
                                    spacings=[0.02],
                                    weights=[1.0, 1.0],
                                    children=[
                                        ContainerNode(
                                            name="equal_bottleneck_row",
                                            layout="row",
                                            spacings=[0.015],
                                            weights=[1.0, 1.0],
                                            children=[
                                                PanelNode("equal_bottleneck", inner_pad=(1.05, 0.35, 0.12, 0.07)),
                                                PanelNode("equal_bottleneck_rank", inner_pad=(0.80, 0.35, 0.12, 0.07)),
                                            ],
                                        ),
                                        ContainerNode(
                                            name="reduced_bottleneck_row",
                                            layout="row",
                                            spacings=[0.015],
                                            weights=[1.0, 1.0],
                                            children=[
                                                PanelNode("reduced_bottleneck", inner_pad=(1.05, 0.55, 0.12, 0.07)),
                                                PanelNode("reduced_bottleneck_rank", inner_pad=(0.80, 0.55, 0.90, 0.07)),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        ContainerNode(
                            name="quentin_results",
                            layout="row",
                            spacings=[0.02],
                            weights=[1.0, 2.0],
                            children=[
                                ContainerNode(
                                    layout="column",
                                    # spacings=[0.00],
                                    # weights=[1.0, 1.2],
                                    children=[
                                        PanelNode("quentin_task_schematic", inner_pad=(0.0, 0.0, 0.0, -0.15)),
                                        # PanelNode("quentin_task_data", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                                    ],
                                ),
                                ContainerNode(
                                    layout="column",
                                    spacings=[0.02],
                                    weights=[1.0, 1.0],
                                    children=[
                                        ContainerNode(
                                            layout="row",
                                            spacings=[0.03],
                                            weights=[1.0, 1.0],
                                            children=[
                                                PanelNode("data_cue_bars", inner_pad=(1.05, 0.15, 0.12, 0.15)),
                                                PanelNode("model_cue_bars", inner_pad=(1.05, 0.15, 0.12, 0.15)),
                                            ],
                                        ),
                                        ContainerNode(
                                            layout="row",
                                            spacings=[0.03],
                                            weights=[1.0, 1.0],
                                            children=[
                                                PanelNode("data_reward_delta_bars", inner_pad=(1.05, 0.15, 0.12, 0.15)),
                                                PanelNode("model_error_delta_bars", inner_pad=(1.05, 0.15, 0.12, 0.15)),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        ContainerNode(
            name="vectorised_instructive_results",
            layout="row",
            spacings=[0.01, 0.01],
            weights=[0.65, 1.0, 1.0],
            children=[
                ContainerNode(
                    layout="column",
                    spacings=[0.02],
                    weights=[1.0, 1.0],
                    children=[
                        PanelNode("vectorised_instructive_task_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                        PanelNode("vectorised_instructive_ndnf_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
                    ],
                ),
                ContainerNode(
                    layout="column",
                    spacings=[0.02],
                    weights=[1.0, 1.0],
                    children=[
                        ContainerNode(
                            layout="row",
                            spacings=[0.01, 0.01],
                            weights=[1.0, 0.73, 1.0],
                            children=[
                                PanelNode("data_decreasing_error", inner_pad=(0.75, 0.50, 0.12, 0.38)),
                                PanelNode("data_increasing_error", inner_pad=(0.05, 0.50, 0.12, 0.38)),
                                PanelNode("data_bar", inner_pad=(0.85, 0.50, 0.12, 0.38)),
                            ],
                        ),
                        ContainerNode(
                            layout="row",
                            spacings=[0.01, 0.01],
                            weights=[1.0, 0.73, 1.0],
                            children=[
                                PanelNode("data_decreasing_error_ndnf", inner_pad=(0.75, 0.50, 0.12, 0.38)),
                                PanelNode("data_increasing_error_ndnf", inner_pad=(0.05, 0.50, 0.12, 0.38)),
                                PanelNode("data_error_bar_ndnf", inner_pad=(0.85, 0.50, 0.12, 0.38)),
                            ],
                        ),
                    ],
                ),
                ContainerNode(
                    layout="column",
                    spacings=[0.02],
                    weights=[1.0, 1.0],
                    children=[
                        ContainerNode(
                            layout="row",
                            spacings=[0.01, 0.01],
                            weights=[1.0, 0.73, 1.0],
                            children=[
                                PanelNode("model_positive_target", inner_pad=(0.75, 0.50, 0.12, 0.38)),
                                PanelNode("model_negative_target", inner_pad=(0.05, 0.50, 0.12, 0.38)),
                                PanelNode("model_bar", inner_pad=(0.85, 0.50, 0.12, 0.38)),
                            ],
                        ),
                        ContainerNode(
                            layout="row",
                            spacings=[0.01, 0.01],
                            weights=[1.0, 0.73, 1.0],
                            children=[
                                PanelNode("model_positive_target_ndnf", inner_pad=(0.75, 0.50, 0.12, 0.38)),
                                PanelNode("model_negative_target_ndnf", inner_pad=(0.05, 0.50, 0.12, 0.38)),
                                PanelNode("model_bar_ndnf", inner_pad=(0.85, 0.50, 0.12, 0.38)),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

fig_manager.resolve_layout(root)
fig_manager.add_spanning_title("dalean_schematic", "Dalean BurstCCN schematic")
fig_manager.add_spanning_title("dalean_ablation_results", "Interneuron silencing effects on plasticity")
fig_manager.add_spanning_title("sst_density_results", "SST interneuron density constrains feedback rank")
fig_manager.add_spanning_title("quentin_results", "Differential cue-error encoding by SST interneurons")#, y_shift=0.12,)
fig_manager.add_spanning_title("vectorised_instructive_results", "Vectorised error signals in cortical dendrites")
fig_manager.insert_pdf("dalean_schematic", "dalean/dalean_schematic_clipped_flipped_Q.pdf")
# fig_manager.insert_pdf("dalean_schematic", "dalean/dalean_schematic_clipped_flipped_Q_tall.pdf")
# fig_manager.insert_pdf("quentin_task_schematic", "dalean/quentin_task_schematic.pdf", align_x='center')
# fig_manager.insert_pdf("quentin_task_data", "dalean/quentin_task_data.pdf", align_x='center')
fig_manager.insert_pdf("quentin_task_schematic", "dalean/quentin_task_schematic_remade2.pdf", align_x='center')
fig_manager.insert_pdf("vectorised_instructive_task_schematic", "dalean/vectorised_instructive_task_remade.pdf", align_x='center', align_y='center')
fig_manager.insert_pdf("vectorised_instructive_ndnf_schematic", "dalean/NDNF_activation2.pdf", align_x='center')

labelled_panels = [
    "dalean_schematic",
    "ablation_data",
    "brain_area_density",
    "equal_bottleneck",
    "reduced_bottleneck",
    "quentin_task_schematic",
    "data_cue_bars",
    # "data_reward_delta_bars",
    "model_cue_bars",
    # "model_error_delta_bars",
    "vectorised_instructive_task_schematic",
    "data_decreasing_error",
    # "data_increasing_error",
    "model_positive_target",
    # "model_negative_target",
    "vectorised_instructive_ndnf_schematic",
    "data_decreasing_error_ndnf",
    "model_positive_target_ndnf",
]
for label, panel_name in zip(ascii_lowercase, labelled_panels):
    fig_manager.add_label(panel_name, label)

ax_ablation_data = fig_manager.create_axes("ablation_data")
ax_ablation_model_weight_change = fig_manager.create_axes("ablation_model_weight_change")
ax_brain_area_density = fig_manager.create_axes("brain_area_density")
ax_equal_bottleneck = fig_manager.create_axes("equal_bottleneck")
ax_equal_bottleneck_rank = fig_manager.create_axes("equal_bottleneck_rank")
ax_reduced_bottleneck = fig_manager.create_axes("reduced_bottleneck")
ax_reduced_bottleneck_rank = fig_manager.create_axes("reduced_bottleneck_rank")

ax_data_cue = fig_manager.create_axes("data_cue_bars")
ax_data_reward = fig_manager.create_axes("data_reward_delta_bars")
ax_model_cue = fig_manager.create_axes("model_cue_bars")
ax_model_error = fig_manager.create_axes("model_error_delta_bars")

ax_data_dec = fig_manager.create_axes("data_decreasing_error")
ax_data_inc = fig_manager.create_axes("data_increasing_error", sharey=ax_data_dec)
ax_data_bar = fig_manager.create_axes("data_bar")
ax_data_dec_ndnf = fig_manager.create_axes("data_decreasing_error_ndnf", sharey=ax_data_dec)
ax_data_inc_ndnf = fig_manager.create_axes("data_increasing_error_ndnf", sharey=ax_data_dec)
ax_data_bar_ndnf = fig_manager.create_axes("data_error_bar_ndnf", sharey=ax_data_bar)
# ax_data_dec_bar = fig_manager.create_axes("data_decreasing_error_bar")
# ax_data_inc_bar = fig_manager.create_axes("data_increasing_error_bar", sharey=ax_data_dec_bar)
ax_model_pos = fig_manager.create_axes("model_positive_target", sharey=ax_data_dec)
ax_model_neg = fig_manager.create_axes("model_negative_target", sharey=ax_data_dec)
ax_model_bar = fig_manager.create_axes("model_bar")
ax_model_pos_ndnf = fig_manager.create_axes("model_positive_target_ndnf", sharey=ax_data_dec)
ax_model_neg_ndnf = fig_manager.create_axes("model_negative_target_ndnf", sharey=ax_data_dec)
ax_model_bar_ndnf = fig_manager.create_axes("model_bar_ndnf", sharey=ax_model_bar)
# ax_model_pos_bar = fig_manager.create_axes("model_positive_target_bar")
# ax_model_neg_bar = fig_manager.create_axes("model_negative_target_bar", sharey=ax_model_pos_bar)

dalean_ablation_plotter = DaleanAblationPlotter()
sst_density_plotter = SSTDensityPlotter()
dalean_bottleneck_plotter = DaleanBottleneckPlotter()
sstp_sstn_plotter = SSTpSSTnPlotter()
vectorised_instructive_plotter = VectorisedInstructivePlotter()

dalean_ablation_plotter.plot_ablation_data(ax_ablation_data, plot_type="bar")
dalean_ablation_plotter.plot_ablation_model(ax_ablation_model_weight_change)

sst_density_plotter.plot_brain_area_density(ax_brain_area_density)
dalean_bottleneck_plotter.plot_equal_bottleneck(ax_equal_bottleneck)
dalean_bottleneck_plotter.plot_equal_rank(ax_equal_bottleneck_rank)
dalean_bottleneck_plotter.plot_reduced_bottleneck(ax_reduced_bottleneck)
dalean_bottleneck_plotter.plot_reduced_rank_heatmap(ax_reduced_bottleneck_rank)

sstp_sstn_plotter.plot_data_cue_bars(ax_data_cue, show_legend=True)
sstp_sstn_plotter.plot_data_reward_delta_bars(ax_data_reward)
sstp_sstn_plotter.plot_model_cue_bars(ax_model_cue)
sstp_sstn_plotter.plot_model_error_delta_bars(ax_model_error)

zscore_model_burst_probability = True

vectorised_instructive_plotter.plot_data_condition(
    ax_data_dec, condition="decreasing_error", use_ndnf_data=False, show_legend=True
)
vectorised_instructive_plotter.plot_data_condition(
    ax_data_inc, condition="increasing_error", use_ndnf_data=False
)
vectorised_instructive_plotter.plot_data_dual_means_bar(
    ax_data_bar,
    use_ndnf_data=False,
)

vectorised_instructive_plotter.plot_data_condition(
    ax_data_dec_ndnf, condition="decreasing_error", use_ndnf_data=True
)
vectorised_instructive_plotter.plot_data_condition(
    ax_data_inc_ndnf, condition="increasing_error", use_ndnf_data=True
)
vectorised_instructive_plotter.plot_data_dual_means_bar(
    ax_data_bar_ndnf,
    use_ndnf_data=True,
)
vectorised_instructive_plotter.plot_model_condition(
    ax_model_pos,
    condition="positive_target",
    use_ndnf_data=False,
    zscore_burst_probability=zscore_model_burst_probability,
)
vectorised_instructive_plotter.plot_model_condition(
    ax_model_neg,
    condition="negative_target",
    use_ndnf_data=False,
    zscore_burst_probability=zscore_model_burst_probability,
)
vectorised_instructive_plotter.plot_model_dual_means_bar(
    ax_model_bar,
    use_ndnf_data=False,
    zscore_burst_probability=zscore_model_burst_probability,
)
vectorised_instructive_plotter.plot_model_condition(
    ax_model_pos_ndnf,
    condition="positive_target",
    use_ndnf_data=True,
    zscore_burst_probability=zscore_model_burst_probability,
)
vectorised_instructive_plotter.plot_model_condition(
    ax_model_neg_ndnf,
    condition="negative_target",
    use_ndnf_data=True,
    zscore_burst_probability=zscore_model_burst_probability,
)
vectorised_instructive_plotter.plot_model_dual_means_bar(
    ax_model_bar_ndnf,
    use_ndnf_data=True,
    zscore_burst_probability=zscore_model_burst_probability,
)
for ax in [ax_data_inc, ax_data_inc_ndnf, ax_model_neg, ax_model_neg_ndnf]:
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("")

fig_manager.add_shared_xlabel([ax_data_dec, ax_data_inc], "SD residual (z-score)")
fig_manager.add_shared_xlabel([ax_data_dec_ndnf, ax_data_inc_ndnf], "SD residual (z-score)")
fig_manager.add_shared_xlabel([ax_model_pos, ax_model_neg], "Burst probability (z-score)")
fig_manager.add_shared_xlabel([ax_model_pos_ndnf, ax_model_neg_ndnf], "Burst probability (z-score)")

superscript_dx = 0.031
index_start = 41
fig_manager.add_panel_group_title_with_superscript(
    ["ablation_data"],
    "Data",
    f"{index_start}",
    underline=False,
    superscript_fontfamily=ref_fontfamily,
    superscript_dx=superscript_dx,
)
fig_manager.add_panel_group_title(["ablation_model_weight_change"], "Model", underline=False)
fig_manager.add_panel_group_title_with_superscript(
    ["brain_area_density"],
    "Data",
    f"{index_start+1},{index_start+2}",
    underline=False,
    superscript_fontfamily=ref_fontfamily,
    superscript_dx=superscript_dx,
)
fig_manager.add_panel_group_title(["equal_bottleneck_row"], "Model", underline=False)
# fig_manager.add_panel_group_title(["reduced_bottleneck_row"], "Model")
fig_manager.add_panel_group_title_with_superscript(
    ["data_cue_bars", "data_reward_delta_bars"],
    "Data",
    f"{index_start+3}",
    underline=False,
    superscript_fontfamily=ref_fontfamily,
    superscript_dx=superscript_dx,
)
fig_manager.add_panel_group_title(["model_cue_bars", "model_error_delta_bars"], "Model", underline=False)
fig_manager.add_panel_group_title_with_superscript(
    ["data_decreasing_error", "data_increasing_error", "data_bar", "data_decreasing_error_ndnf", "data_increasing_error_ndnf", "data_error_bar_ndnf"],
    "Data",
    f"{index_start+4}",
    underline=False,
    superscript_fontfamily=ref_fontfamily,
    superscript_dx=superscript_dx,
)
fig_manager.add_panel_group_title(
    ["model_positive_target", "model_negative_target", "model_bar", "model_positive_target_ndnf", "model_negative_target_ndnf", "model_bar_ndnf"],
    "Model",
    underline=False,
)

fig_manager.finalise_figure("dalean.pdf", draw_debug=False, show=False)
