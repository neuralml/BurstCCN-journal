from matplotlib import pyplot as plt

from plotting.figures.figure_manager import FigureManager, ContainerNode, PanelNode
from plotting.plotters.dalean_ablation import DaleanAblationPlotter
from plotting.utils import init_global_matplotlib_constants

init_global_matplotlib_constants()

plot_width = 12
aspect_ratio = 0.32

fig = plt.figure(figsize=(plot_width, plot_width * aspect_ratio), constrained_layout=False)
fig_manager = FigureManager(fig, margin_inch=(0.45, 0.25, 0.15, 0.35))

root = ContainerNode(
    name="dalean_ablation_sm",
    layout="row",
    spacings=[0.05, 0.05],
    weights=[1.0, 1.0, 1.0],
    children=[
        PanelNode("ablation_model_weight_change", inner_pad=(0.95, 0.60, 0.18, 0.10)),
        PanelNode("ablation_model_burst_probability_change", inner_pad=(0.95, 0.60, 0.18, 0.10)),
        PanelNode("ablation_model_event_rate_change", inner_pad=(0.95, 0.60, 0.18, 0.10)),
    ],
)

fig_manager.resolve_layout(root)

fig_manager.add_label("ablation_model_weight_change", "a")
fig_manager.add_label("ablation_model_burst_probability_change", "b")
fig_manager.add_label("ablation_model_event_rate_change", "c")

ax_weight_change = fig_manager.create_axes("ablation_model_weight_change")
ax_burst_probability_change = fig_manager.create_axes("ablation_model_burst_probability_change")
ax_event_rate_change = fig_manager.create_axes("ablation_model_event_rate_change")

dalean_ablation_plotter = DaleanAblationPlotter()
dalean_ablation_plotter.plot_ablation_model(ax_weight_change)
dalean_ablation_plotter.plot_ablation_model_burst_probability_change(ax_burst_probability_change)
dalean_ablation_plotter.plot_ablation_model_event_rate_change(ax_event_rate_change)

fig_manager.finalise_figure("dalean_sm.pdf", draw_debug=False, show=False)
