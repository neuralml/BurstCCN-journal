from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from plotting.analysis.fmnist_dalean_bottleneck import DaleanBottleneckRankResultsStore, DaleanBottleneckResultsStore
from plotting.plot_specs.plot_specs_base import PlotColours
from plotting.plot_specs.dalean_bottleneck import DaleanBottleneckAxDetailsStore, DaleanBottleneckElemDetailsStore
from plotting.plotters.plotter_base import run_plots
from plotting.utils import setup_axis


class DaleanBottleneckPlotter:
    def __init__(self):
        self.results = None
        self.rank_results = DaleanBottleneckRankResultsStore()
        self.metric_key = "epoch/top1_error_best/test"
        self.ax_details = DaleanBottleneckAxDetailsStore()
        self.elem_details = DaleanBottleneckElemDetailsStore()

    def _get_results(self):
        if self.results is None:
            self.results = DaleanBottleneckResultsStore()
        return self.results

    def _get_step_key(self):
        is_batch_data = self.metric_key.startswith("batch")
        results = self._get_results()
        step_key = results.BATCH_EPOCH_KEY if is_batch_data else results.EPOCH_KEY
        return step_key, is_batch_data

    def plot_equal_bottleneck(self, ax):
        setup_axis(ax, **self.ax_details.get("equal_bottleneck").to_kwargs())
        curve_meta = self.elem_details.get("test_error_curve")
        ref_meta = self.elem_details.get("sst50_ref")

        group = "equal"
        results = self._get_results()
        run_params = results.get_group_params(group)
        bottleneck_sizes = run_params["bottleneck_sizes"]
        step_key, is_batch_data = self._get_step_key()

        final_test_errors = []
        final_test_sems = []
        for bottleneck_size in bottleneck_sizes:
            run_filter = results.get_wandb_run_filter(group, bottleneck_size=bottleneck_size)
            _, mean, sem = results.fetch_and_summarise(
                **run_filter,
                step_key=step_key,
                data_key=self.metric_key,
                batch_to_epoch=is_batch_data,
                final_only=True,
            )
            final_test_errors.append(mean[0] if len(mean) > 0 else float("nan"))
            final_test_sems.append(sem[0] if len(sem) > 0 else float("nan"))

        x = [size for size in bottleneck_sizes]
        y = [err for _, err in zip(bottleneck_sizes, final_test_errors)]
        yerr = [err for _, err in zip(bottleneck_sizes, final_test_sems)]

        if 50 in x:
            idx_50 = x.index(50)
            y_50 = y[idx_50]
            x_plot = [xi for i, xi in enumerate(x) if i != idx_50]
            y_plot = [yi for i, yi in enumerate(y) if i != idx_50]
            yerr_plot = [ei for i, ei in enumerate(yerr) if i != idx_50]
        else:
            y_50 = None
            x_plot, y_plot, yerr_plot = x, y, yerr

        ax.errorbar(
            x_plot,
            y_plot,
            yerr=yerr_plot,
            linestyle=curve_meta.line_style,
            marker=curve_meta.marker_style,
            capsize=4,
            color=curve_meta.line_colour,
        )
        ax.set_xticks(x_plot)
        ax.set_xticklabels([str(size) for size in x_plot])

        if y_50 is not None:
            ax.axhline(y_50, color=ref_meta.line_colour, linestyle=ref_meta.line_style)
            # ax.text(
            #     0.98,
            #     y_50,
            #     ref_meta.display_name,
            #     transform=ax.get_yaxis_transform(),
            #     ha="right",
            #     va="bottom",
            #     color=ref_meta.line_colour,
            #     fontsize=9,
            # )

    def plot_reduced_bottleneck(self, ax):
        setup_axis(ax, **self.ax_details.get("reduced_bottleneck").to_kwargs())
        curve_meta = self.elem_details.get("test_error_curve")

        group = "reduced"
        results = self._get_results()
        run_params = results.get_group_params(group)
        reduction_layer_names = run_params["reduction_layer"]
        reduction_layers = list(range(1, len(reduction_layer_names) + 1))
        step_key, is_batch_data = self._get_step_key()

        final_test_errors = []
        final_test_sems = []
        for reduction_layer_name in reduction_layer_names:
            run_filter = results.get_wandb_run_filter(group, reduction_layer=reduction_layer_name)
            _, mean, sem = results.fetch_and_summarise(
                **run_filter,
                step_key=step_key,
                data_key=self.metric_key,
                batch_to_epoch=is_batch_data,
                final_only=True,
            )
            final_test_errors.append(mean[0] if len(mean) > 0 else float("nan"))
            final_test_sems.append(sem[0] if len(sem) > 0 else float("nan"))

        x = [size for size in reduction_layers]
        y = [err for _, err in zip(reduction_layers, final_test_errors)]
        yerr = [err/2 for _, err in zip(reduction_layers, final_test_sems)]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            linestyle=curve_meta.line_style,
            marker=curve_meta.marker_style,
            capsize=4,
            color=curve_meta.line_colour,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in x])

    def plot_equal_rank(self, ax):
        results = self.rank_results.get_equal_apical_pr()
        all_layers = sorted({layer for result in results for layer in result["metrics_by_layer"]})
        x_values = [result["scan_value"] for result in results]
        x_plot_values = [value for value in x_values if value != 50]

        for layer in all_layers[::-1]:
            layer_colour = PlotColours.from_layer_index(layer)
            xs = []
            ys = []
            yerr = []
            y_50 = None
            for result in results:
                value = result["metrics_by_layer"].get(layer, {}).get("fa_pr")
                if value is None or np.isnan(value):
                    continue
                if result["scan_value"] == 50:
                    y_50 = value
                    continue
                xs.append(result["scan_value"])
                ys.append(value)
                yerr.append(result["sem_by_layer"].get(layer, {}).get("fa_pr", 0.0))
            if ys:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=layer_colour,
                    label=f"Area {layer + 1}",
                )
            # if y_50 is not None:
            #     ax.axhline(y_50, color=layer_colour, linestyle="--", linewidth=1.2)

        ax.set_xlabel("# SST")
        ax.set_ylabel("SST feedback rank")
        ax.set_xlim(0.0, 22.0)
        ax.set_xticks(x_plot_values)
        ax.legend(loc="lower right", bbox_to_anchor=(1.10, 0.03), borderaxespad=0.0)

    def plot_reduced_rank_heatmap(self, ax):
        data = self.rank_results.get_reduced_layer_apical_pr_heatmap()
        heatmap = data["heatmap"]
        finite_values = heatmap[np.isfinite(heatmap)]
        vmin = float(finite_values.min()) if finite_values.size else 0.0
        cmap = LinearSegmentedColormap.from_list("baseline_drop", ["#b30000", "#ffffff"])

        image = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=100.0,
            cmap=cmap,
        )
        ax.set_xlabel("Reduced SST area")
        ax.set_ylabel("Measured SST area")
        ax.set_xticks(range(len(data["reduced_layers"])))
        ax.set_xticklabels([str(layer) for layer in data["reduced_layers"]])
        ax.set_yticks(range(len(data["measured_layers"])))
        ax.set_yticklabels([str(layer) for layer in data["measured_layers"]])
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Fraction of\noriginal rank (%)")

    def plot_reduced_angle_fa_by_layer(self, ax):
        results = self._get_results()
        layer_indices = list(range(4))

        equal_8_filter = results.get_wandb_run_filter("equal", bottleneck_size=8)
        equal_8_layer_means = self._get_angle_fa_layer_means(equal_8_filter, layer_indices)

        run_specs = [
            ("Equal 8", equal_8_filter),
        ]
        reduced_params = results.get_group_params("reduced")
        run_specs.extend(
            (reduction_layer_name, results.get_wandb_run_filter("reduced", reduction_layer=reduction_layer_name))
            for reduction_layer_name in reduced_params["reduction_layer"]
        )

        for label, run_filter in run_specs:
            layer_means = self._get_angle_fa_layer_means(run_filter, layer_indices)
            angle_differences = layer_means - equal_8_layer_means
            ax.plot(
                layer_indices,
                angle_differences,
                marker="o",
                linewidth=1.8,
                label=label,
            )

        ax.set_xlabel("Area index")
        ax.set_ylabel("Model FA-update angle - original FA-update angle (deg)")
        ax.set_xticks(layer_indices)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, zorder=0)
        ax.legend(title="Reduced area", frameon=False)

    def plot_reduced_angle_fa_by_reduced_layer(self, ax):
        self._plot_reduced_angle_by_reduced_layer_heatmap(ax, angle_name="angle_fa", label="ANN-FA")

    def plot_reduced_angle_bp_by_reduced_layer(self, ax):
        self._plot_reduced_angle_by_reduced_layer_heatmap(ax, angle_name="angle_bp", label="ANN-BP")

    def plot_reduced_angle_wy_by_reduced_layer(self, ax):
        self._plot_reduced_angle_by_reduced_layer_heatmap(ax, angle_name="angle_WY", label="WY", clip_negative=False)

    def plot_reduced_angle_fa_by_reduced_layer_lines(self, ax):
        self._plot_reduced_angle_by_reduced_layer_heatmap(ax, angle_name="angle_fa", label="ANN-FA", plot_type="line")

    def plot_reduced_angle_bp_by_reduced_layer_lines(self, ax):
        self._plot_reduced_angle_by_reduced_layer_heatmap(ax, angle_name="angle_bp", label="ANN-BP", plot_type="line")

    def plot_reduced_angle_wy_by_reduced_layer_lines(self, ax):
        self._plot_reduced_angle_by_reduced_layer_heatmap(
            ax,
            angle_name="angle_WY",
            label="WY",
            clip_negative=False,
            plot_type="line",
        )

    def _plot_reduced_angle_by_reduced_layer_heatmap(
            self,
            ax,
            angle_name,
            label,
            clip_negative=True,
            plot_type="heatmap",
    ):
        results = self._get_results()
        layer_indices = list(range(4))
        reduced_params = results.get_group_params("reduced")
        reduction_layer_names = reduced_params["reduction_layer"]
        reduction_layers = list(range(1, len(reduction_layer_names) + 1))

        equal_8_filter = results.get_wandb_run_filter("equal", bottleneck_size=8)
        equal_8_seed_means = self._get_angle_layer_seed_means(equal_8_filter, layer_indices, angle_name=angle_name)

        differences_by_reduction = []
        sem_by_reduction = []
        for reduction_layer_name in reduction_layer_names:
            run_filter = results.get_wandb_run_filter("reduced", reduction_layer=reduction_layer_name)
            seed_means = self._get_angle_layer_seed_means(run_filter, layer_indices, angle_name=angle_name)
            diff_mean, diff_sem = self._get_angle_layer_difference_mean_sem(
                seed_means,
                equal_8_seed_means,
                layer_indices,
                angle_name,
            )
            differences_by_reduction.append(diff_mean)
            sem_by_reduction.append(diff_sem)

        heatmap = np.asarray(differences_by_reduction, dtype=float).T
        sem_by_reduction = np.asarray(sem_by_reduction, dtype=float).T
        if clip_negative:
            heatmap = np.clip(heatmap, 0.0, None)

        if plot_type == "line":
            for layer_idx in layer_indices:
                ax.errorbar(
                    reduction_layers,
                    heatmap[layer_idx],
                    yerr=sem_by_reduction[layer_idx],
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=PlotColours.from_layer_index(layer_idx),
                    label=f"Area {layer_idx + 1}",
                )

            ax.set_xlabel("Reduced area")
            ax.set_ylabel(f"{label} angle - original {label} angle (deg)")
            ax.set_xticks(reduction_layers)
            ax.set_xticklabels([str(layer) for layer in reduction_layers])
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, zorder=0)
            ax.legend(title="Measured area", frameon=False)
            return
        if plot_type != "heatmap":
            raise ValueError(f"Unknown plot_type {plot_type!r}. Expected 'heatmap' or 'line'.")

        finite_values = heatmap[np.isfinite(heatmap)]
        if clip_negative:
            vmin = 0.0
            vmax = float(finite_values.max()) if finite_values.size else 1.0
            cmap = LinearSegmentedColormap.from_list("angle_increase", ["#ffffff", "#b30000"])
            norm = None
        else:
            max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
            vmin = -max_abs
            vmax = max_abs
            cmap = "RdBu_r"
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        image_kwargs = {
            "aspect": "auto",
            "origin": "lower",
            "interpolation": "nearest",
            "cmap": cmap,
        }
        if norm is None:
            image_kwargs.update({"vmin": vmin, "vmax": vmax})
        else:
            image_kwargs["norm"] = norm

        image = ax.imshow(heatmap, **image_kwargs)
        ax.set_xlabel("Reduced area")
        ax.set_ylabel("Measured area")
        ax.set_xticks(range(len(reduction_layers)))
        ax.set_xticklabels([str(layer) for layer in reduction_layers])
        ax.set_yticks(range(len(layer_indices)))
        ax.set_yticklabels([str(layer + 1) for layer in layer_indices])
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
            f"{label} angle -\noriginal {label} angle (deg)"
        )

    def plot_equal_angle_fa_by_sst(self, ax):
        self._plot_equal_angle_by_sst(ax, angle_name="angle_fa", ylabel="ANN-FA angle (deg)")

    def plot_equal_angle_bp_by_sst(self, ax):
        self._plot_equal_angle_by_sst(ax, angle_name="angle_bp", ylabel="ANN-BP angle (deg)")

    def plot_equal_angle_wy_by_sst(self, ax):
        self._plot_equal_angle_by_sst(ax, angle_name="angle_WY", ylabel="WY angle (deg)")

    def _plot_equal_angle_by_sst(self, ax, angle_name, ylabel):
        results = self._get_results()
        layer_indices = list(range(4))
        bottleneck_sizes = results.get_group_params("equal")["bottleneck_sizes"]

        means_by_bottleneck = []
        sems_by_bottleneck = []
        for bottleneck_size in bottleneck_sizes:
            run_filter = results.get_wandb_run_filter("equal", bottleneck_size=bottleneck_size)
            mean, sem = self._get_angle_layer_mean_sem(run_filter, layer_indices, angle_name=angle_name)
            means_by_bottleneck.append(mean)
            sems_by_bottleneck.append(sem)

        means_by_bottleneck = np.asarray(means_by_bottleneck, dtype=float)
        sems_by_bottleneck = np.asarray(sems_by_bottleneck, dtype=float)
        plot_indices = [idx for idx, size in enumerate(bottleneck_sizes) if size != 50]
        x_values = [bottleneck_sizes[idx] for idx in plot_indices]
        ref_idx = bottleneck_sizes.index(50) if 50 in bottleneck_sizes else None

        for layer_idx in layer_indices:
            layer_colour = PlotColours.from_layer_index(layer_idx)
            ax.errorbar(
                x_values,
                means_by_bottleneck[plot_indices, layer_idx],
                yerr=sems_by_bottleneck[plot_indices, layer_idx],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=layer_colour,
                label=f"Area {layer_idx + 1}",
            )
            if ref_idx is not None:
                ax.axhline(
                    means_by_bottleneck[ref_idx, layer_idx],
                    color=layer_colour,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=0,
                )

        ax.set_xlabel("# SST")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_values)
        ax.legend(title="Measured area", frameon=False)

    def _get_angle_fa_layer_means(self, run_filter, layer_indices):
        return self._get_angle_layer_means(run_filter, layer_indices, angle_name="angle_fa")

    def _get_angle_layer_means(self, run_filter, layer_indices, angle_name):
        mean, _ = self._get_angle_layer_mean_sem(run_filter, layer_indices, angle_name)
        return mean

    def _get_angle_layer_mean_sem(self, run_filter, layer_indices, angle_name):
        seed_means = self._get_angle_layer_seed_means(run_filter, layer_indices, angle_name)
        angle_keys = [f"batch/{angle_name}/layer_{layer_idx}" for layer_idx in layer_indices]
        values = seed_means[angle_keys].to_numpy(dtype=float)
        return self._mean_sem(values, axis=0)

    def _get_angle_layer_seed_means(self, run_filter, layer_indices, angle_name):
        results = self._get_results()
        angle_keys = [f"batch/{angle_name}/layer_{layer_idx}" for layer_idx in layer_indices]
        data = results.fetch(
            **run_filter,
            keys=[results.BATCH_KEY, *angle_keys],
            sort_by=results.BATCH_KEY,
        )
        return data.groupby("seed", as_index=False)[angle_keys].mean()

    def _get_angle_layer_difference_mean_sem(self, seed_means, baseline_seed_means, layer_indices, angle_name):
        angle_keys = [f"batch/{angle_name}/layer_{layer_idx}" for layer_idx in layer_indices]
        merged = seed_means.merge(
            baseline_seed_means,
            on="seed",
            how="inner",
            suffixes=("", "_baseline"),
        )
        differences = np.column_stack([
            merged[key].to_numpy(dtype=float) - merged[f"{key}_baseline"].to_numpy(dtype=float)
            for key in angle_keys
        ])
        return self._mean_sem(differences, axis=0)

    @staticmethod
    def _mean_sem(values, axis=0):
        values = np.asarray(values, dtype=float)
        mean = np.nanmean(values, axis=axis)
        n = np.sum(np.isfinite(values), axis=axis)
        std = np.nanstd(values, axis=axis, ddof=1)
        sem = std / np.sqrt(np.maximum(n, 1))
        sem = np.where(n > 1, sem, 0.0)
        return mean, sem


if __name__ == "__main__":
    PLOT_REGISTRY = {
        "equal_rank": {
            "fn": lambda p, ax: p.plot_equal_rank(ax),
            "figsize": (5, 4),
        },
        "reduced_rank_heatmap": {
            "fn": lambda p, ax: p.plot_reduced_rank_heatmap(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_fa_by_layer": {
            "fn": lambda p, ax: p.plot_reduced_angle_fa_by_layer(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_fa_by_reduced_layer": {
            "fn": lambda p, ax: p.plot_reduced_angle_fa_by_reduced_layer(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_bp_by_reduced_layer": {
            "fn": lambda p, ax: p.plot_reduced_angle_bp_by_reduced_layer(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_wy_by_reduced_layer": {
            "fn": lambda p, ax: p.plot_reduced_angle_wy_by_reduced_layer(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_fa_by_reduced_layer_lines": {
            "fn": lambda p, ax: p.plot_reduced_angle_fa_by_reduced_layer_lines(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_bp_by_reduced_layer_lines": {
            "fn": lambda p, ax: p.plot_reduced_angle_bp_by_reduced_layer_lines(ax),
            "figsize": (5, 4),
        },
        "reduced_angle_wy_by_reduced_layer_lines": {
            "fn": lambda p, ax: p.plot_reduced_angle_wy_by_reduced_layer_lines(ax),
            "figsize": (5, 4),
        },
        "equal_angle_fa_by_sst": {
            "fn": lambda p, ax: p.plot_equal_angle_fa_by_sst(ax),
            "figsize": (5, 4),
        },
        "equal_angle_bp_by_sst": {
            "fn": lambda p, ax: p.plot_equal_angle_bp_by_sst(ax),
            "figsize": (5, 4),
        },
        "equal_angle_wy_by_sst": {
            "fn": lambda p, ax: p.plot_equal_angle_wy_by_sst(ax),
            "figsize": (5, 4),
        },
    }

    run_plots(
        DaleanBottleneckPlotter,
        PLOT_REGISTRY,
        plot_names=[
            "reduced_angle_fa_by_reduced_layer_lines",
            "reduced_angle_bp_by_reduced_layer_lines",
            "reduced_angle_wy_by_reduced_layer_lines",
            "equal_angle_fa_by_sst",
            "equal_angle_bp_by_sst",
            "equal_angle_wy_by_sst",
        ],
    )
