import matplotlib.lines as mlines

from plotting.analysis.image_tasks import ImageTaskResultsStore
from plotting.plot_specs.image_tasks import ImageTaskElemDetailsStore, ImageTaskAxDetailsStore
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis, plot_line


class ImageTaskPlotter:
    def __init__(self):
        self.results = ImageTaskResultsStore()

        self.elem_details = ImageTaskElemDetailsStore()
        self.ax_details = ImageTaskAxDetailsStore()

    def _plot_batch_metric(self, ax, run_filter, metric_key, **plot_kwargs):
        batch_key = self.results.BATCH_KEY
        batches, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                              step_key=batch_key,
                                                              data_key=metric_key)
        # examples = batches * 32
        return plot_line(ax, batches, mean, sem, **plot_kwargs)

    def _plot_epoch_metric(self, ax, run_filter, metric_key, **plot_kwargs):
        epoch_key = self.results.EPOCH_KEY
        batch_epoch_key = self.results.BATCH_EPOCH_KEY

        is_batch_data = metric_key.startswith('batch')
        step_key = batch_epoch_key if is_batch_data else epoch_key

        epochs, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                             step_key=step_key,
                                                             data_key=metric_key,
                                                             batch_to_epoch=is_batch_data)

        line = plot_line(ax, epochs, mean, sem, **plot_kwargs)
        return epochs, mean, sem, line

    def plot_top1_test_performance(
        self,
        ax,
        task="cifar10",
        model_type=None,
        ax_name="top1_test_performance",
        show_legend=False,
        smoothing_alpha=0.4,
        legend_titles=None,
        legend_locations=None,
        legend_bbox_to_anchor=None,
    ):
        metric_key = self.results.TOP1_TEST_ERROR_KEY
        self._plot_metric_modes(
            ax,
            ax_name,
            metric_key,
            task,
            model_type=model_type,
            show_legend=show_legend,
            smoothing_alpha=smoothing_alpha,
            legend_titles=legend_titles,
            legend_locations=legend_locations,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

    def plot_top5_test_performance(
        self,
        ax,
        task="cifar10",
        model_type=None,
        ax_name="top5_test_performance",
        show_legend=False,
        smoothing_alpha=0.4,
        legend_titles=None,
        legend_locations=None,
        legend_bbox_to_anchor=None,
    ):
        metric_key = self.results.TOP5_TEST_ERROR_KEY
        self._plot_metric_modes(
            ax,
            ax_name,
            metric_key,
            task,
            model_type=model_type,
            show_legend=show_legend,
            smoothing_alpha=smoothing_alpha,
            legend_titles=legend_titles,
            legend_locations=legend_locations,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

    def plot_BP_align(
        self,
        ax,
        task="cifar10",
        model_type=None,
        ax_name="BP_align",
        show_legend=False,
        smoothing_alpha=0.2,
    ):
        metric_key = self.results.ANGLE_KEYS['bp']
        self._plot_metric_modes(
            ax,
            ax_name,
            metric_key,
            task,
            model_type=model_type,
            show_legend=show_legend,
            smoothing_alpha=smoothing_alpha,
        )

    def plot_WY_align(self, ax, task="cifar10", model_type=None, ax_name="WY_align", show_legend=False):
        metric_key = self.results.ANGLE_KEYS['wy']
        self._plot_metric_modes(ax, ax_name, metric_key, task, model_type=model_type, show_legend=show_legend)

    def _plot_metric_modes(
        self,
        ax,
        ax_name,
        metric_key,
        task,
        model_type=None,
        show_legend=False,
        smoothing_alpha=None,
        legend_titles=None,
        legend_locations=None,
        legend_bbox_to_anchor=None,
    ):
        group_params = self.results.get_group_params(task)

        model_types = group_params['model_types'] if model_type is None else [model_type]
        modes = group_params['modes']

        for mode in modes:
            mode_meta = self.elem_details.get(mode)
            mode_kwargs = mode_meta.to_kwargs()
            mode_line_style = mode_kwargs.get("line_style", "-")
            for model_type in model_types:
                model_meta = self.elem_details.get(model_type)
                model_kwargs = model_meta.to_kwargs()
                run_filter = self.results.get_wandb_run_filter(task, model_type=model_type, mode=mode)
                epochs, mean, sem, _ = self._plot_epoch_metric(
                    ax,
                    run_filter,
                    metric_key,
                    line_colour=model_kwargs.get("line_colour"),
                    line_style=mode_line_style,
                    smoothing_alpha=smoothing_alpha,
                )

                if metric_key in {self.results.TOP1_TEST_ERROR_KEY, self.results.TOP5_TEST_ERROR_KEY}:
                    print(
                        f"[ImageTaskPlotter] task={task} model_type={model_type} mode={mode} "
                        f"metric={metric_key} final_epoch={epochs[-1]} "
                        f"mean={mean[-1]:.6f}, stderr={sem[-1]:.6f}",
                        flush=True,
                    )

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

        if show_legend:
            legend_titles = legend_titles or {}
            legend_locations = legend_locations or {}
            legend_bbox_to_anchor = legend_bbox_to_anchor or {}
            mode_handles = []
            for mode in modes:
                mode_meta = self.elem_details.get(mode)
                mode_kwargs = mode_meta.to_kwargs()
                label = mode_kwargs.get("display_name", mode)
                line_style = mode_kwargs.get("line_style", "-")
                mode_handles.append(mlines.Line2D([], [], color='black', linestyle=line_style, label=label))

            if mode_handles:
                mode_legend = ax.legend(
                    handles=mode_handles,
                    title=legend_titles.get("mode"),
                    loc=legend_locations.get("mode", "center right"),
                    bbox_to_anchor=legend_bbox_to_anchor.get("mode"),
                    fontsize=9,
                    title_fontsize=10,
                    handlelength=1.6,
                    borderaxespad=0.0,
                )
                ax.add_artist(mode_legend)

            model_type_handles = []
            for model_type in model_types:
                model_meta = self.elem_details.get(model_type)
                model_kwargs = model_meta.to_kwargs()
                display_name = model_kwargs.get("display_name", model_type)
                line_colour = model_kwargs.get("line_colour", "black")
                model_type_handles.append(mlines.Line2D([], [], color=line_colour, linestyle='solid', label=display_name))

            if model_type_handles:
                model_legend = ax.legend(
                    handles=model_type_handles,
                    title=legend_titles.get("model"),
                    loc=legend_locations.get("model", "upper right"),
                    bbox_to_anchor=legend_bbox_to_anchor.get("model"),
                    fontsize=9,
                    title_fontsize=10,
                    handlelength=1.6,
                    borderaxespad=0.0,
                )


if __name__ == "__main__":
    task = 'imagenet'
    model_type = 'burstccn'

    PLOT_REGISTRY = {
        "top1_test_performance": {
            "fn": lambda p, ax, **kw: p.plot_top1_test_performance(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"task": task, "model_type": model_type, "show_legend": True}
        },
        "top5_test_performance": {
            "fn": lambda p, ax, **kw: p.plot_top5_test_performance(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"task": task, "model_type": model_type, "show_legend": True}
        },
        "BP_align": {
            "fn": lambda p, ax, **kw: p.plot_BP_align(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"task": task, "model_type": model_type}
        },
        "WY_align": {
            "fn": lambda p, ax, **kw: p.plot_WY_align(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"task": task, "model_type": model_type}
        },
    }

    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY)
    # run_plots(ImageTaskPlotter, PLOT_REGISTRY, "top1_test_performance")
    run_plots(ImageTaskPlotter, PLOT_REGISTRY, "top5_test_performance")
    run_plots(ImageTaskPlotter, PLOT_REGISTRY, "BP_align")
    run_plots(ImageTaskPlotter, PLOT_REGISTRY, "WY_align")
    # run_plots(ImageTaskPlotter, PLOT_REGISTRY, "FA_align")
