from matplotlib import pyplot as plt

from plotting.plotters.continuous_burstccn import ContinuousBurstCCNPlotter

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 15.0
aspect_ratio = 0.55

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio))

fig_manager = FigureManager(fig)

root = ContainerNode(layout="row", weights=[0.65, 1], spacings=[0.025], children=[
    PanelNode("model_schematic"),
    ContainerNode(name='results', layout="column", weights=[0.45, 1, 0.75, 1], spacings=[0.035, 0.035, 0.035], children=[
        PanelNode("sin_task_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
        ContainerNode(name='sin_results', layout="row", spacings=[0.01, 0.00], children=[
            PanelNode("sin_loss", inner_pad=(0.85, 0.65, 0.2, 0.05)),
            PanelNode("sin_outputs", inner_pad=(0.75, 0.65, 0.2, 0.05)),
            PanelNode("sin_bp", inner_pad=(1.15, 0.65, 0.2, 0.05))
        ]),
        PanelNode("catcam_task_schematic", inner_pad=(0.0, 0.0, 0.0, 0.0)),
        ContainerNode(name='catcam_results', layout="row", spacings=[0.01, 0.00], children=[
            PanelNode("catcam_loss", inner_pad=(0.85, 0.65, 0.4, 0.05)),
            PanelNode("catcam_outputs", inner_pad=(0.85, 0.65, 0.2, 0.05)),
            PanelNode("catcam_bp", inner_pad=(1.15, 0.65, 0.2, 0.05))
        ]),
    ])
])

# root = ContainerNode(layout="row", weights=[0.65, 1], spacings=[0.025], children=[
#     PanelNode("model_schematic"),
#     ContainerNode(name='results', layout="column", weights=[1, 1], spacings=[0.05], children=[
#         ContainerNode(name='sin_task', layout="column", weights=[1, 1], spacings=[0.035], children=[
#             ContainerNode(layout="row", weights=[1, 1], spacings=[0.01], children=[
#                 PanelNode("sin_task_schematic"),
#                 PanelNode("sin_loss", inner_pad=(0.85, 0.65, 0.2, 0.05)),
#             ]),
#             ContainerNode(name='sin_results', layout="row", weights=[1, 1], spacings=[0.01], children=[
#                 PanelNode("sin_outputs", inner_pad=(0.75, 0.65, 0.2, 0.05)),
#                 PanelNode("sin_bp", inner_pad=(1.15, 0.65, 0.2, 0.05)),
#             ]),
#         ]),
#         ContainerNode(name='catcam_task', layout="column", weights=[1, 1], spacings=[0.035], children=[
#             ContainerNode(layout="row", weights=[1, 1], spacings=[0.01], children=[
#                 PanelNode("catcam_task_schematic"),
#                 PanelNode("catcam_loss", inner_pad=(0.85, 0.65, 0.4, 0.05)),
#             ]),
#             ContainerNode(name='catcam_results', layout="row", weights=[1, 1], spacings=[0.01], children=[
#                 PanelNode("catcam_outputs", inner_pad=(0.85, 0.65, 0.2, 0.05)),
#                 PanelNode("catcam_bp", inner_pad=(1.15, 0.65, 0.2, 0.05)),
#             ]),
#         ]),
#     ])
# ])

fig_manager.resolve_layout(root)

# fig_manager.add_panel("model_schematic", left=0.0, bottom=0.0, right=0.4, top=1.0)
fig_manager.add_label("model_schematic", "a")
fig_manager.add_label("sin_task_schematic", "b")
fig_manager.add_label("sin_results", "c")
fig_manager.add_label("catcam_task_schematic", "d")
fig_manager.add_label("catcam_results", "e")

fig_manager.insert_pdf("model_schematic", "continuous_model/continuous_model_schematic_v3.pdf")
fig_manager.insert_pdf("sin_task_schematic", "continuous_model/sin_wave_task_v4.pdf")
fig_manager.insert_pdf("catcam_task_schematic", "continuous_model/catcam_task_v5.pdf")

sin_loss_ax = fig_manager.create_axes("sin_loss")
sin_outputs_ax = fig_manager.create_axes("sin_outputs")
sin_bp_ax = fig_manager.create_axes("sin_bp")

catcam_loss_ax = fig_manager.create_axes("catcam_loss")
catcam_outputs_ax = fig_manager.create_axes("catcam_outputs")
catcam_bp_ax = fig_manager.create_axes("catcam_bp")

plotter = ContinuousBurstCCNPlotter()

task = 'sin_wave'
seed = 4
start_step = 1
end_step = 200001
# end_step = 200001
neuron_id = 0
start = 1100
end = 2100

plotter.plot_loss(sin_loss_ax, task=task, block_size=1000, seeds=[seed])
plotter.plot_output_comparison(sin_outputs_ax, task=task, start_step=start_step, end_step=end_step, neuron_id=neuron_id,
                               start=start, end=end, seed=seed)
plotter.plot_burst_probability(sin_bp_ax, task=task, start_step=start_step, end_step=end_step, neuron_id=neuron_id,
                               start=start, end=end, seed=seed)

task = 'catcam'
seed = 1
start_step = 1
end_step = 10500001
neuron_id = 7
start = 400
end = 3400

plotter.plot_loss(catcam_loss_ax, task=task, block_size=1000, seeds=[seed])
plotter.plot_output_comparison(catcam_outputs_ax, task=task, start_step=start_step, end_step=end_step, neuron_id=neuron_id,
                               start=start, end=end, seed=seed)
plotter.plot_burst_probability(catcam_bp_ax, task=task, start_step=start_step, end_step=end_step, neuron_id=neuron_id,
                               start=start, end=end, seed=seed)

fig_manager.finalise_figure("continuous_burstccn.pdf", draw_debug=False, show=False)
