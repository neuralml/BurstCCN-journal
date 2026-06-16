from pathlib import Path

import numpy as np
import matplotlib.ticker as mticker

from plotting.analysis.mnist import MNISTApicalActivityResultsStore, MNISTResultsStore
from plotting.analysis.results_store_base import build_per_seed_function
from plotting.plot_specs.mnist import MNISTApicalActivityElemDetailsStore, MNISTApicalActivityAxDetailsStore, \
    MNISTElemDetailsStore, MNISTAxDetailsStore
from plotting.plot_specs.plot_specs_base import PlotLabels
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis, plot_line, rescale_ticks, plot_scatter


class MNISTApicalActivityPlotter:
    def __init__(self, h5_path=None, neuron_index=4, branch_index=0):
        if h5_path is None:
            h5_path = (
                Path(__file__).resolve().parents[2]
                / "saved_activities"
                / "mnist_burstccn_Y_learning_rand_rand_with_teacher-iw54li9b.h5"
            )

        self.results = MNISTApicalActivityResultsStore(
            h5_path=h5_path,
            neuron_index=neuron_index,
            branch_index=branch_index,
        )

        self.elem_details = MNISTApicalActivityElemDetailsStore()
        self.ax_details = MNISTApicalActivityAxDetailsStore()

    def plot_apical_QY_scatter_before(self, ax, **kwargs):
        self._plot_apical_QY_scatter(ax, when='before', **kwargs)

    def plot_apical_QY_scatter_after(self, ax, **kwargs):
        self._plot_apical_QY_scatter(ax, when='after', **kwargs)

    def _plot_apical_QY_scatter(self, ax, ax_name="QY_scatter", when='before', show_legend=True, **kwargs):
        Q_input, Y_input = self.results.get_Q_Y_inputs(100, when)

        QY_scatter_meta = self.elem_details.get('QY_scatter')
        plot_scatter(ax, Q_input, Y_input, **QY_scatter_meta.to_kwargs())

        xmin, xmax = ax.get_xlim()
        line_x = np.array([xmin, xmax])

        QY_scatter_equal_line_meta = self.elem_details.get('QY_scatter_equal_line')
        plot_line(ax, line_x, -line_x, **QY_scatter_equal_line_meta.to_kwargs())

        ax.set_title(f"{when.capitalize()} plasticity", fontfamily="Consolas", color="black", fontsize=15, pad=6)
        # ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=1, alpha=1.0, zorder=0)
        ax.axvline(0, color="black", linewidth=1, alpha=1.0, zorder=0)

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

        if show_legend:
            ax.legend()

    def plot_apical_exc_inh_inputs_before(self, ax, **kwargs):
        self._plot_apical_exc_inh_inputs(ax, when='before', **kwargs)

    def plot_apical_exc_inh_inputs_after(self, ax, **kwargs):
        self._plot_apical_exc_inh_inputs(ax, when='after', **kwargs)

    def _plot_apical_exc_inh_inputs(self, ax, ax_name="exc_inh_inputs", when='before', show_legend=True, **kwargs):
        n_samples = 32
        x_line = np.arange(1, n_samples + 1)

        exc_input, inh_input, total_input = self.results.get_exc_inh_inputs(n_samples, when)

        exc_input_line_meta = self.elem_details.get('exc_input')
        inh_input_line_meta = self.elem_details.get('inh_input')
        total_input_line_meta = self.elem_details.get('total_input')

        plot_line(ax, x_line, exc_input, **exc_input_line_meta.to_kwargs())
        plot_line(ax, x_line, inh_input, **inh_input_line_meta.to_kwargs())
        plot_line(ax, x_line, total_input, **total_input_line_meta.to_kwargs())

        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())
        if show_legend:
            # ax.legend(loc="center right", bbox_to_anchor=(1.65, 0.5),)
            ax.legend(loc="upper left", bbox_to_anchor=(-0.4, 0.75))


class MNISTPlotter:
    def __init__(self):
        self.results = MNISTResultsStore()

        self.elem_details = MNISTElemDetailsStore()
        self.ax_details = MNISTAxDetailsStore()

    def _plot_batch_metric(self, ax, run_filter, metric_key, **plot_kwargs):
        batch_key = self.results.BATCH_KEY
        batches, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                              step_key=batch_key,
                                                              data_key=metric_key)
        # examples = batches * 32
        return plot_line(ax, batches, mean, sem, **plot_kwargs)

    def _plot_epoch_metric(self, ax, run_filter, metric_keys, metric_fn=None, **plot_kwargs):
        epoch_key = self.results.EPOCH_KEY
        batch_epoch_key = self.results.BATCH_EPOCH_KEY

        resolved_metric_keys = [metric_keys] if isinstance(metric_keys, str) else list(metric_keys)
        if not resolved_metric_keys:
            raise ValueError("'metric_keys' must contain at least one metric key.")

        is_batch_data = resolved_metric_keys[0].startswith('batch')
        for key in resolved_metric_keys[1:]:
            if key.startswith('batch') != is_batch_data:
                raise ValueError("All metric keys must be either batch metrics or epoch metrics.")
        step_key = batch_epoch_key if is_batch_data else epoch_key

        out_key = resolved_metric_keys[0] if len(resolved_metric_keys) == 1 else "combined_metric"
        if metric_fn is None:
            metric_key = resolved_metric_keys[0]
            metric_fn = lambda g: g[metric_key]

        per_seed_fn = build_per_seed_function(
            step_key=step_key,
            out_key=out_key,
            fn=metric_fn,
        )

        epochs, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                             step_key=step_key,
                                                             data_keys=resolved_metric_keys,
                                                             batch_to_epoch=is_batch_data,
                                                             per_seed_fn=per_seed_fn,
                                                             out_key=out_key)
        line = plot_line(ax, epochs, mean, sem, **plot_kwargs)
        return epochs, mean, sem, line

    def plot_QY_align_across_branches(self, ax, ax_name="QY_across_branches"):
        angle_key = self.results.ANGLE_KEYS['qy']

        group = 'Y_learning_branches'
        group_params = self.results.get_group_params(group)

        n_branches_list = group_params['n_branches']
        for n_branches in n_branches_list:
            run_filter = self.results.get_wandb_run_filter(group, n_branches=n_branches)
            # batches, mean, sem = self.results.fetch_and_summarise(**run_filter,
            #                                                       step_key=batch_key,
            #                                                       data_key=angle_key)
            #
            # examples = batches * 32
            # plot_line(ax, examples, mean, sem, display_name=n_branches)

            green_map = {
                1: "#74C476",
                2: "#41AB5D",
                5: "#238B45",
                10: "#006D2C",
                15: "#00441B",
            }

            colour = green_map[n_branches]
            self._plot_batch_metric(ax, run_filter, angle_key, line_colour=colour, display_name=n_branches)

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())
        ax.legend(
            title="Apical\nbranches",
            loc="center right",
            bbox_to_anchor=(1.35, 0.5),
        )
        rescale_ticks(ax, axis='x', order=5)

    def plot_QY_align_across_noise(self, ax, ax_name="QY_across_noise"):
        angle_key = self.results.ANGLE_KEYS['qy']

        group = 'Y_learning_forward_noise'
        group_params = self.results.get_group_params(group)

        noise_list = group_params['noise']
        for noise in noise_list:
            run_filter = self.results.get_wandb_run_filter(group, noise=noise)
            # batches, mean, sem = self.results.fetch_and_summarise(**run_filter,
            #                                                       step_key=batch_key,
            #                                                       data_key=angle_key)
            #
            # examples = batches * 32
            # plot_line(ax, examples, mean, sem, display_name=noise)
            noise_name = noise if noise != 'null' else '0.0'

            blue_map = {
                'null': "#6BAED6",
                0.05: "#4292C6",
                0.1: "#2171B5",
                0.2: "#08519C",
                0.4: "#08306B",
            }
            colour = blue_map[noise]

            self._plot_batch_metric(ax, run_filter, angle_key, line_colour=colour, display_name=noise_name)

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())
        ax.legend(
            title="Noise std",
            loc="center right",
            bbox_to_anchor=(1.35, 0.5),
        )
        rescale_ticks(ax, axis='x', order=5)

    def plot_QY_align_across_error_scale(self, ax, ax_name="QY_across_error_scale"):
        angle_key = self.results.ANGLE_KEYS['qy']

        group = 'Y_learning_error_scale'
        group_params = self.results.get_group_params(group)

        error_scale_list = group_params['error_scale']
        for error_scale in error_scale_list[::-1]:
            run_filter = self.results.get_wandb_run_filter(group, error_scale=error_scale)
            # batches, mean, sem = self.results.fetch_and_summarise(**run_filter,
            #                                                       step_key=batch_key,
            #                                                       data_key=angle_key)
            #
            # examples = batches * 32
            # plot_line(ax, examples, mean, sem, display_name=error_scale)
            # red_map = {
            #     1.0: "#FB6A4A",
            #     0.75: "#EF3B2C",
            #     0.5: "#CB181D",
            #     0.25: "#A50F15",
            #     0.0: "#67000D",
            # }

            red_map = {
                0.0: "#FB6A4A",
                0.25: "#EF3B2C",
                0.5: "#CB181D",
                0.75: "#A50F15",
                1.0: "#67000D",
            }

            colour = red_map[error_scale]
            self._plot_batch_metric(ax, run_filter, angle_key, line_colour=colour, display_name=error_scale)

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())
        ax.legend(
            title="Teacher\nstrength",
            loc="center right",
            bbox_to_anchor=(1.35, 0.5),
        )
        rescale_ticks(ax, axis='x', order=5)

    def plot_QY_vs_FA_align_scatter(self, ax, ax_name="QY_vs_FA_align_scatter", dataset="branches"):
        qy_key = self.results.ANGLE_KEYS['qy']
        fa_key = self.results.ANGLE_KEYS['fa']
        batch_key = self.results.BATCH_KEY
        n_bins = 10
        n_per_bin = 19
        rng = np.random.default_rng(0)

        dataset_specs = {
            "branches": {
                "group": "Y_learning_branches",
                "param_name": "n_branches",
                "title": "Apical branches",
                "colours": {
                    1: "#74C476",
                    2: "#41AB5D",
                    5: "#238B45",
                    10: "#006D2C",
                    15: "#00441B",
                },
            },
            "noise": {
                "group": "Y_learning_forward_noise",
                "param_name": "noise",
                "title": "Forward noise",
                "colours": {
                    'null': "#6BAED6",
                    0.05: "#4292C6",
                    0.1: "#2171B5",
                    0.2: "#08519C",
                    0.4: "#08306B",
                },
            },
            "error_scale": {
                "group": "Y_learning_error_scale",
                "param_name": "error_scale",
                "title": "Teacher strength",
                "colours": {
                    0.0: "#FB6A4A",
                    0.25: "#EF3B2C",
                    0.5: "#CB181D",
                    0.75: "#A50F15",
                    1.0: "#67000D",
                },
            },
        }
        if dataset not in dataset_specs:
            raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(dataset_specs)}.")

        spec = dataset_specs[dataset]
        group = spec["group"]
        param_name = spec["param_name"]

        group_params = self.results.get_group_params(group)
        for param_value in group_params[param_name]:
            if dataset == "error_scale" and param_value == 0.0:
                continue

            run_filter = self.results.get_wandb_run_filter(group, **{param_name: param_value})
            _, qy_mean, _ = self.results.fetch_and_summarise(
                **run_filter,
                step_key=batch_key,
                data_key=qy_key,
            )
            _, fa_mean, _ = self.results.fetch_and_summarise(
                **run_filter,
                step_key=batch_key,
                data_key=fa_key,
            )

            qy_vals = np.asarray(qy_mean[1:]).ravel()
            fa_vals = np.asarray(fa_mean[1:]).ravel()
            n_points = min(qy_vals.size, fa_vals.size)
            qy_vals = qy_vals[:n_points]
            fa_vals = fa_vals[:n_points]
            valid = np.isfinite(qy_vals) & np.isfinite(fa_vals) & (qy_vals > 0) & (fa_vals > 0)
            if not np.any(valid):
                continue
            qy_vals, fa_vals = self._sample_by_log_x_bins(
                qy_vals[valid],
                fa_vals[valid],
                n_bins=n_bins,
                n_per_bin=n_per_bin,
                rng=rng,
            )
            if qy_vals.size == 0:
                continue

            param_label = "0.0" if param_value == "null" else param_value
            ax.scatter(
                qy_vals,
                fa_vals,
                s=9,
                alpha=0.35,
                marker="o",
                color=spec["colours"][param_value],
                linewidths=0,
                label=param_label,
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        plain_number_formatter = mticker.FuncFormatter(lambda val, _: f"{val:g}")
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_yticks([20, 30, 40, 50, 60, 70, 80, 90])
        ax.xaxis.set_major_formatter(plain_number_formatter)
        ax.yaxis.set_major_formatter(plain_number_formatter)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.set_xlabel(PlotLabels.QY_ALIGNMENT)
        ax.set_ylabel(PlotLabels.FA_ALIGNMENT)
        ax.legend(title=spec["title"], fontsize=7, frameon=False, loc="best")

    @staticmethod
    def _sample_by_log_x_bins(x, y, n_bins=10, n_per_bin=100, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        log_x = np.log10(x)
        if log_x.min() == log_x.max():
            idx = np.arange(x.size)
            if idx.size > n_per_bin:
                idx = rng.choice(idx, size=n_per_bin, replace=False)
            return x[idx], y[idx]

        bins = np.linspace(log_x.min(), log_x.max(), n_bins + 1)
        keep = []

        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if i == n_bins - 1:
                mask = (log_x >= lo) & (log_x <= hi)
            else:
                mask = (log_x >= lo) & (log_x < hi)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            if idx.size > n_per_bin:
                idx = rng.choice(idx, size=n_per_bin, replace=False)
            keep.append(idx)

        if not keep:
            return np.array([]), np.array([])

        idx = np.concatenate(keep)
        return x[idx], y[idx]

    def plot_Y_learning_apical_magnitude(self, ax, ax_name="Y_learning_apical_magnitude"):
        metric_key = self.results.APICAL_MAGNITUDE_KEY
        self._plot_Y_only_learning_metric(ax, ax_name, metric_key=metric_key)

    def plot_Y_learning_burst_prob_magnitude(self, ax, ax_name="Y_learning_burst_prob_magnitude"):
        metric_key = self.results.BURST_PROB_MAGNITUDE_KEY
        self._plot_Y_only_learning_metric(ax, ax_name, metric_key=metric_key)

    def plot_Y_learning_angle_fa(self, ax, ax_name="Y_learning_angle_fa"):
        metric_key = self.results.ANGLE_KEYS['fa']
        self._plot_Y_only_learning_metric(ax, ax_name, metric_key=metric_key)

    def _plot_Y_only_learning_metric(self, ax, ax_name, metric_key):
        group = "Y_learning_no_teacher"

        run_filter = self.results.get_wandb_run_filter(group)

        line_meta = self.elem_details.get('Y_only_learning_metric')
        self._plot_batch_metric(ax, run_filter, metric_key, **line_meta.to_kwargs())

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

        rescale_ticks(ax, axis='x', order=5)

    def plot_Y_learning_apical_magnitude_scatter(self, ax, ax_name="Y_learning_apical_magnitude_scatter"):
        qy_key = self.results.ANGLE_KEYS['qy']
        # apic_key = self.results.APICAL_MAGNITUDE_KEY
        apic_key = 'batch/burst_prob_change_variance/global'
        # apic_key = self.results.ANGLE_KEYS['fa']

        subsample_frac = 0.2

        batch_key = self.results.BATCH_KEY
        groups = [
            ("Y_learning_branches", "n_branches"),
            # ("Y_learning_error_scale", "error_scale"),
            ("Y_learning_forward_noise", "noise"),
        ]

        apic_means = []
        qy_means = []

        for group, param_name in groups:
            group_params = self.results.get_group_params(group)
            param_values = group_params[param_name]
            for param_value in param_values:
                run_filter = self.results.get_wandb_run_filter(group, **{param_name: param_value})
                _, apic_mean, _ = self.results.fetch_and_summarise(
                    **run_filter,
                    step_key=batch_key,
                    data_key=apic_key,
                )
                _, qy_mean, _ = self.results.fetch_and_summarise(
                    **run_filter,
                    step_key=batch_key,
                    data_key=qy_key,
                )
                # qy_mean = np.log(qy_mean)

                apic_means.append(np.asarray(apic_mean[1:]).ravel())
                qy_means.append(np.asarray(qy_mean[1:]).ravel())

        # cos_qy = np.cos(np.deg2rad(qy_mean))

        def stratified_subsample_x(x, y, n_bins=50, n_per_bin=200, rng=None):
            if rng is None:
                rng = np.random.default_rng()

            bins = np.linspace(x.min(), x.max(), n_bins + 1)
            keep_x = []
            keep_y = []

            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (x >= lo) & (x < hi)
                idx = np.flatnonzero(mask)

                if idx.size == 0:
                    continue

                if idx.size > n_per_bin:
                    idx = rng.choice(idx, size=n_per_bin, replace=False)

                keep_x.append(x[idx])
                keep_y.append(y[idx])

            return np.concatenate(keep_x), np.concatenate(keep_y)

        if qy_means and apic_means:
            qy_vals = np.concatenate(qy_means)
            apic_vals = np.concatenate(apic_means)
            qy_vals, apic_vals = stratified_subsample_x(
                qy_vals,
                apic_vals,
                n_bins=10,
                n_per_bin=150,
            )
            ax.scatter(qy_vals, apic_vals, s=1.5)
            # ax.hexbin(qy_vals, apic_vals)


        ax.set_yscale('log')
        # ax.set_xscale('log')

        # ax_metadata = self.ax_details.get(ax_name)
        # setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_with_without_Y_learning_test_performance(self, ax, ax_name="with_without_Y_learning_test_performance",
                                                      **kwargs):
        metric_key = self.results.TEST_ERROR_KEY
        self._plot_with_without_Y_learning(ax, ax_name, [metric_key], **kwargs)

    def plot_with_without_Y_learning_QY_align(self, ax, ax_name="with_without_Y_learning_QY_align", **kwargs):
        metric_key = self.results.ANGLE_KEYS['qy']
        self._plot_with_without_Y_learning(ax, ax_name, [metric_key], **kwargs)

    def plot_with_without_Y_learning_BP_align(self, ax, ax_name="with_without_Y_learning_BP_align", **kwargs):
        metric_key = self.results.ANGLE_KEYS['bp']
        self._plot_with_without_Y_learning(ax, ax_name, [metric_key], **kwargs)

    def plot_with_without_Y_learning_FA_align(self, ax, ax_name="with_without_Y_learning_FA_align", **kwargs):
        # metric_key = self.results.ANGLE_KEYS['fa']
        # metric_key = "batch/angle_fa/global_average"
        metric_keys = [
            'batch/angle_fa/layer_2',
            'batch/angle_fa/layer_1',
            'batch/angle_fa/layer_0',
        ]
        metric_fn = lambda g: g[metric_keys].mean(axis=1)

        self._plot_with_without_Y_learning(ax, ax_name, metric_keys, metric_fn=metric_fn, **kwargs)

    def _plot_with_without_Y_learning(self, ax, ax_name, metric_keys, metric_fn=None, show_legend=True, **kwargs):
        group = 'fa_performance_with_without_Y_learning'
        group_params = self.results.get_group_params(group)
        Y_lrs = group_params['Y_lrs']

        for Y_lr in Y_lrs:
            run_filter = self.results.get_wandb_run_filter(group, Y_lr=Y_lr)
            line_meta_key = 'Y_learning_on' if Y_lr != 0.0 else 'Y_learning_off'
            line_meta = self.elem_details.get(line_meta_key)
            epochs, mean, sem, _ = self._plot_epoch_metric(
                ax,
                run_filter,
                metric_keys,
                metric_fn=metric_fn,
                **line_meta.to_kwargs(),
            )

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

        if show_legend:
            ax.legend(title="Y plasticity", loc='center right')

    def plot_across_apical_branches_test_error(self, ax, ax_name='across_apical_branches_test_error'):
        # metric_key = self.results.BEST_TEST_ERROR_KEY
        metric_key = self.results.TEST_ERROR_KEY
        # metric_key = 'epoch/top1_error/train'

        self._plot_across_apical_branches_metric(ax, ax_name, metric_key=metric_key)

    def plot_across_apical_branches_QY_align(self, ax, ax_name='across_apical_branches_QY_align'):
        metric_key = self.results.ANGLE_KEYS['qy']
        self._plot_across_apical_branches_metric(ax, ax_name, metric_key=metric_key)

    def plot_across_apical_branches_FA_align(self, ax, ax_name='across_apical_branches_FA_align'):
        metric_key = self.results.ANGLE_KEYS['fa']
        self._plot_across_apical_branches_metric(ax, ax_name, metric_key=metric_key)

    def plot_across_apical_branches_BP_align(self, ax, ax_name='across_apical_branches_BP_align'):
        metric_key = self.results.ANGLE_KEYS['bp']
        self._plot_across_apical_branches_metric(ax, ax_name, metric_key=metric_key)

    def _plot_across_apical_branches_metric(self, ax, ax_name, metric_key):
        group = 'fa_performance_branches'
        group_params = self.results.get_group_params(group)
        n_branches_list = group_params['n_branches']

        means = []
        sems = []

        for n_branches in n_branches_list:
            run_filter = self.results.get_wandb_run_filter(group, n_branches=n_branches)

            epoch_key = self.results.EPOCH_KEY
            batch_epoch_key = self.results.BATCH_EPOCH_KEY

            is_batch_data = metric_key.startswith('batch')
            step_key = batch_epoch_key if is_batch_data else epoch_key

            epochs, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                                 step_key=step_key,
                                                                 data_key=metric_key,
                                                                 batch_to_epoch=is_batch_data,
                                                                 final_only=True)

            means.append(mean.item())
            sems.append(sem.item())

        means = np.array(means)
        sems = np.array(sems)

        line_meta = self.elem_details.get('across_apical_branches_metric')
        plot_line(ax, n_branches_list, means, sems, **line_meta.to_kwargs())

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, x_ticks=n_branches_list, **ax_metadata.to_kwargs())

    def plot_without_teacher_blocks_test_error(self, ax, ax_name='without_teacher_blocks_test_error'):
        metric_key = self.results.BEST_TEST_ERROR_KEY
        self._plot_without_teacher_blocks_metric(ax, ax_name, metric_key=metric_key)

    def plot_without_teacher_blocks_QY_align(self, ax, ax_name='without_teacher_blocks_QY_align'):
        metric_key = self.results.ANGLE_KEYS['qy']
        self._plot_without_teacher_blocks_metric(ax, ax_name, metric_key=metric_key)

    def plot_without_teacher_blocks_BP_align(self, ax, ax_name='without_teacher_blocks_BP_align'):
        metric_key = self.results.ANGLE_KEYS['bp']
        self._plot_without_teacher_blocks_metric(ax, ax_name, metric_key=metric_key)

    def plot_without_teacher_blocks_FA_align(self, ax, ax_name='without_teacher_blocks_FA_align'):
        metric_key = self.results.ANGLE_KEYS['fa']
        self._plot_without_teacher_blocks_metric(ax, ax_name, metric_key=metric_key)

    def _plot_without_teacher_blocks_metric(self, ax, ax_name, metric_key):
        epoch_key = self.results.EPOCH_KEY
        batch_epoch_key = self.results.BATCH_EPOCH_KEY
        is_batch_data = metric_key.startswith('batch')
        step_key = batch_epoch_key if is_batch_data else epoch_key

        group = 'fa_performance_block_training'
        group_params = self.results.get_group_params(group)
        n_block_batches_list = group_params['n_block_batches']

        means = []
        sems = []

        for n_block_batches in n_block_batches_list:
            run_filter = self.results.get_wandb_run_filter(group, n_block_batches=n_block_batches)

            epochs, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                                 step_key=step_key,
                                                                 data_key=metric_key,
                                                                 batch_to_epoch=is_batch_data,
                                                                 final_only=True)

            means.append(mean.item())
            sems.append(sem.item())

        means = np.array(means)
        sems = np.array(sems)

        line_meta = self.elem_details.get('without_teacher_blocks_metric')
        ratios_list = [b / 1500 for b in n_block_batches_list]
        plot_line(ax, ratios_list, means, sems, **line_meta.to_kwargs())

        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_network_depth_test_errors(self, ax, ax_name='network_depth_test_errors', show_legend=False):
        metric_key = self.results.BEST_TEST_ERROR_KEY
        self._plot_network_depth_metric(ax, ax_name, metric_key=metric_key, show_legend=show_legend)

    def plot_network_depth_BP_align(self, ax, ax_name='network_depth_BP_align', show_legend=False):
        # metric_key = self.results.ANGLE_KEYS['bp']
        # metric_key = "batch/angle_bp/global"
        metric_key = "batch/angle_bp/layer_0"
        metric_fn = lambda g: g[metric_key].iloc[:50].mean()

        self._plot_network_depth_metric(ax, ax_name, metric_key=metric_key, metric_fn=metric_fn,
                                        show_legend=show_legend)

    def plot_network_depth_FA_align(self, ax, ax_name='network_depth_FA_align', show_legend=False):
        # metric_key = self.results.ANGLE_KEYS['fa']
        # metric_key = "batch/angle_fa/global_hidden"
        metric_key = "batch/angle_fa/layer_0"
        metric_fn = lambda g: g[metric_key].iloc[:50].mean()

        self._plot_network_depth_metric(ax, ax_name, metric_key=metric_key, metric_fn=metric_fn,
                                        show_legend=show_legend)

    def _plot_network_depth_metric(self, ax, ax_name, metric_key, metric_fn=None, show_legend=False):
        epoch_key = self.results.EPOCH_KEY
        batch_epoch_key = self.results.BATCH_EPOCH_KEY
        is_batch_data = metric_key.startswith('batch')
        step_key = batch_epoch_key if is_batch_data else epoch_key
        if metric_fn is None:
            metric_fn = lambda g: g[metric_key]

        group = 'fa_performance'
        group_params = self.results.get_group_params(group)
        model_types = group_params["model_types"]
        n_hidden_layers_list = group_params["n_hidden_layers"]

        lines = []
        for model_type in model_types:
            means = []
            sems = []
            for n_hidden_layers in n_hidden_layers_list:
                run_filter = self.results.get_wandb_run_filter(group, model_type=model_type,
                                                               n_hidden_layers=n_hidden_layers)

                out_key = f"{metric_key}_mean"
                per_seed_fn = build_per_seed_function(
                    step_key=step_key,
                    out_key=out_key,
                    fn=metric_fn,
                )

                epochs, mean, sem = self.results.fetch_and_summarise(**run_filter,
                                                                     step_key=step_key,
                                                                     data_key=metric_key,
                                                                     batch_to_epoch=is_batch_data,
                                                                     final_only=True,
                                                                     per_seed_fn=per_seed_fn,
                                                                     out_key=out_key)

                if metric_key in {self.results.BEST_TEST_ERROR_KEY, self.results.TEST_ERROR_KEY} and n_hidden_layers == 4:
                    print(
                        f"[MNISTPlotter] network_depth_test_errors "
                        f"model_type={model_type} n_hidden_layers={n_hidden_layers}: "
                        f"mean={mean.item():.6f}, stderr={sem.item():.6f}",
                        flush=True,
                    )
                means.append(mean.item())
                sems.append(sem.item())

            means = np.array(means)
            sems = np.array(sems)

            line_meta = self.elem_details.get(model_type)
            line = plot_line(ax, n_hidden_layers_list, means, sems, **line_meta.to_kwargs(), marker_style='o')

            lines.append(line)

        ax_meta = self.ax_details.get(ax_name)
        setup_axis(ax, x_ticks=n_hidden_layers_list, **ax_meta.to_kwargs())
        if show_legend:
            # Original position with all models: ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),)
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12),)

        # if ax_meta.y_scale == 'log':
        #     import matplotlib.ticker as mticker
        #     ax.yaxis.set_major_formatter(
        #         mticker.FuncFormatter(lambda y, _: f'{y:g}')
        #     )
        #     ax.yaxis.set_minor_formatter(
        #         mticker.FuncFormatter(lambda y, _: f'{y:g}')
        #     )


if __name__ == "__main__":
    APICAL_PLOT_REGISTRY = {
        "apical_QY_scatter_before": {
            "fn": lambda p, ax, **kw: p.plot_apical_QY_scatter_before(ax, **kw),
            "figsize": (4, 3),
            # "kwargs": {"when": "before"},
        },
        "apical_QY_scatter_after": {
            "fn": lambda p, ax, **kw: p.plot_apical_QY_scatter_after(ax, **kw),
            "figsize": (4, 3),
            # "kwargs": {"when": "after"},
        },
        "apical_exc_inh_inputs_before": {
            "fn": lambda p, ax, **kw: p.plot_apical_exc_inh_inputs_before(ax, **kw),
            "figsize": (4, 2.5),
            # "kwargs": {"when": "before"},
        },
        "apical_exc_inh_inputs_after": {
            "fn": lambda p, ax, **kw: p.plot_apical_exc_inh_inputs_after(ax, **kw),
            "figsize": (4, 2.5),
            # "kwargs": {"when": "after"},
        },
    }

    MNIST_PLOT_REGISTRY = {
        "QY_align_across_branches": {
            "fn": lambda p, ax, **kw: p.plot_QY_align_across_branches(ax, **kw),
            "figsize": (4, 3)
        },
        "QY_align_across_noise": {
            "fn": lambda p, ax, **kw: p.plot_QY_align_across_noise(ax, **kw),
            "figsize": (4, 3)
        },
        "QY_align_across_error_scale": {
            "fn": lambda p, ax, **kw: p.plot_QY_align_across_error_scale(ax, **kw),
            "figsize": (4, 3)
        },
        "QY_vs_FA_align_scatter_branches": {
            "fn": lambda p, ax, **kw: p.plot_QY_vs_FA_align_scatter(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"dataset": "branches"},
        },
        "QY_vs_FA_align_scatter_noise": {
            "fn": lambda p, ax, **kw: p.plot_QY_vs_FA_align_scatter(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"dataset": "noise"},
        },
        "QY_vs_FA_align_scatter_error_scale": {
            "fn": lambda p, ax, **kw: p.plot_QY_vs_FA_align_scatter(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"dataset": "error_scale"},
        },
        "Y_learning_apical_magnitude": {
            "fn": lambda p, ax, **kw: p.plot_Y_learning_apical_magnitude(ax, **kw),
            "figsize": (4, 3)
        },
        "Y_learning_angle_fa": {
            "fn": lambda p, ax, **kw: p.plot_Y_learning_angle_fa(ax, **kw),
            "figsize": (4, 3)
        },
        "with_without_Y_learning_test_performance": {
            "fn": lambda p, ax, **kw: p.plot_with_without_Y_learning_test_performance(ax, **kw),
            "figsize": (4, 3)
        },
        "with_without_Y_learning_QY_align": {
            "fn": lambda p, ax, **kw: p.plot_with_without_Y_learning_QY_align(ax, **kw),
            "figsize": (4, 3)
        },
        "with_without_Y_learning_BP_align": {
            "fn": lambda p, ax, **kw: p.plot_with_without_Y_learning_BP_align(ax, **kw),
            "figsize": (4, 3)
        },
        "across_apical_branches_test_error": {
            "fn": lambda p, ax, **kw: p.plot_across_apical_branches_test_error(ax, **kw),
            "figsize": (4, 3),
        },
        "across_apical_branches_QY_align": {
            "fn": lambda p, ax, **kw: p.plot_across_apical_branches_QY_align(ax, **kw),
            "figsize": (4, 3),
        },
        "across_apical_branches_FA_align": {
            "fn": lambda p, ax, **kw: p.plot_across_apical_branches_FA_align(ax, **kw),
            "figsize": (4, 3),
        },
        "across_apical_branches_BP_align": {
            "fn": lambda p, ax, **kw: p.plot_across_apical_branches_BP_align(ax, **kw),
            "figsize": (4, 3),
        },
        "without_teacher_blocks_test_error": {
            "fn": lambda p, ax, **kw: p.plot_without_teacher_blocks_test_error(ax, **kw),
            "figsize": (4, 3),
        },
        "without_teacher_blocks_QY_align": {
            "fn": lambda p, ax, **kw: p.plot_without_teacher_blocks_QY_align(ax, **kw),
            "figsize": (4, 3),
        },
        "without_teacher_blocks_BP_align": {
            "fn": lambda p, ax, **kw: p.plot_without_teacher_blocks_BP_align(ax, **kw),
            "figsize": (4, 3),
        },
        "network_depth_test_errors": {
            "fn": lambda p, ax, **kw: p.plot_network_depth_test_errors(ax, **kw),
            "figsize": (4, 3),
            "kwargs": {"show_legend": True}
        },
        "network_depth_BP_align": {
            "fn": lambda p, ax, **kw: p.plot_network_depth_BP_align(ax, **kw),
            "figsize": (4, 3),
        },
        "network_depth_FA_align": {
            "fn": lambda p, ax, **kw: p.plot_network_depth_FA_align(ax, **kw),
            "figsize": (4, 3),
        },

        "Y_learning_apical_magnitude_scatter": {
            "fn": lambda p, ax, **kw: p.plot_Y_learning_apical_magnitude_scatter(ax, **kw),
            "figsize": (4, 3)
        },

    }

    # run_plots(MNISTApicalActivityPlotter, APICAL_PLOT_REGISTRY)
    # run_plots(MNISTApicalActivityPlotter, APICAL_PLOT_REGISTRY, "apical_QY_scatter_before")
    # run_plots(MNISTApicalActivityPlotter, APICAL_PLOT_REGISTRY, "apical_QY_scatter_after")
    # run_plots(MNISTApicalActivityPlotter, APICAL_PLOT_REGISTRY, "apical_exc_inh_inputs_before")
    # run_plots(MNISTApicalActivityPlotter, APICAL_PLOT_REGISTRY, "apical_exc_inh_inputs_after")


    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY)

    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "QY_align_across_branches")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "QY_align_across_noise")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "QY_align_across_error_scale")

    run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, [
        "QY_vs_FA_align_scatter_branches",
        "QY_vs_FA_align_scatter_noise",
        "QY_vs_FA_align_scatter_error_scale",
    ])
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "with_without_Y_learning_test_performance")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "with_without_Y_learning_QY_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "with_without_Y_learning_BP_align")
    #
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "across_apical_branches_test_error")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "across_apical_branches_QY_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "across_apical_branches_FA_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "across_apical_branches_BP_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "without_teacher_blocks_test_error")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "without_teacher_blocks_test_error")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "without_teacher_blocks_QY_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "without_teacher_blocks_BP_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "network_depth_test_errors")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "network_depth_BP_align")
    # run_plots(MNISTPlotter, MNIST_PLOT_REGISTRY, "network_depth_FA_align")
