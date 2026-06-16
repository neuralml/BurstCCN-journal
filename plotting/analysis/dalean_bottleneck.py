from matplotlib import pyplot as plt

from plotting.analysis.results_store_base import WandbResultsStore
from plotting.utils import plot_line, setup_axis


class DaleanBottleneckResultsStore(WandbResultsStore):
    def __init__(self):
        super().__init__(cache_path='mnist')

        self.EPOCH_KEY = "epoch"
        self.BATCH_EPOCH_KEY = "batch/epoch"

        self.TOP1_TEST_ERROR_KEY = "epoch/top1_error/test"
        self.BEST_TOP1_TEST_ERROR_KEY = "epoch/top1_error_best/test"

        self.TOP5_TEST_ERROR_KEY = "epoch/top5_error/test"
        self.BEST_TOP5_TEST_ERROR_KEY = "epoch/top5_error_best/test"

        self.BATCH_KEY = "batch"
        self.ANGLE_KEYS = {"qy": "batch/angle_QY/global",
                           "fa": "batch/angle_fa/global_hidden",
                           "bp": "batch/angle_bp/global_hidden",
                           "wy": "batch/angle_WY/global"
                           }

    def get_group_params(self, group):
        GROUP_PARAM_SETS = {
            "equal": dict(
                # bottleneck_sizes=[5, 7, 9, 10, 11, 15, 30, 50, 100, 200],
                # bottleneck_sizes=[5, 7, 10, 15, 30, 50, 100, 200],
                bottleneck_sizes=[5, 7, 10, 15, 30, 50, 500],
            ),
        }

        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "equal":
            bottleneck_size = kwargs['bottleneck_size']
            if bottleneck_size == 500:
                run_name = f"mnist_burstccn_dales_5h_full"
            else:
                run_name = f"mnist_burstccn_dales_5h_eq{bottleneck_size}"
        else:
            raise ValueError(f"Invalid group: {group}")

        return run_name

    def get_wandb_group_name(self, group, **kwargs):
        group_dict = {
            'equal': 'mnist_burstccn_dales_tmp',
        }

        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        wandb_run_name = self.get_wandb_run_name(group, **kwargs)
        wandb_group = self.get_wandb_group_name(group, **kwargs)

        return {"run_name": wandb_run_name,
                "group": wandb_group}


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axis(ax, x_label="# SST", y_label="Test error (%)")

    res = DaleanBottleneckResultsStore()

    group = 'equal'
    run_params = res.get_group_params(group)

    bottleneck_sizes = run_params['bottleneck_sizes']

    metric_key = 'epoch/top1_error_best/test'
    epoch_key = res.EPOCH_KEY
    batch_epoch_key = res.BATCH_EPOCH_KEY
    is_batch_data = metric_key.startswith('batch')
    step_key = batch_epoch_key if is_batch_data else epoch_key

    final_test_errors = []
    final_test_sems = []
    for bottleneck_size in bottleneck_sizes:
        run_filter = res.get_wandb_run_filter(group, bottleneck_size=bottleneck_size)
        print(run_filter)
        _, mean, sem = res.fetch_and_summarise(
            **run_filter,
            step_key=step_key,
            data_key=metric_key,
            batch_to_epoch=is_batch_data,
            final_only=True,
        )
        final_test_errors.append(mean[0] if len(mean) > 0 else float("nan"))
        final_test_sems.append(sem[0] if len(sem) > 0 else float("nan"))

    left_sizes = [size for size in bottleneck_sizes if size < 500]
    left_errors = [err for size, err in zip(bottleneck_sizes, final_test_errors) if size < 500]
    left_sems = [err for size, err in zip(bottleneck_sizes, final_test_sems) if size < 500]

    plot_line(ax, left_sizes, left_errors, left_sems, marker_style="o")
    ax.set_xlim(0, max(left_sizes) * 1.1)
    ax.set_xticks(left_sizes)
    ax.set_xticklabels([str(size) for size in left_sizes])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if 500 in bottleneck_sizes:
        idx_500 = bottleneck_sizes.index(500)
        err_500 = final_test_errors[idx_500]
        ax.axhline(err_500, linestyle="--", color="black", linewidth=1.2)
        ax.text(ax.get_xlim()[0]+2, err_500, "100% SST", ha="left", va="bottom", fontsize=10)
    fig.tight_layout()
    plt.show()
