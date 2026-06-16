from plotting.analysis.results_store_base import WandbResultsStore



class CIFARResultsStore(WandbResultsStore):
    def __init__(self):
        super().__init__(cache_path='mnist')

        self.EPOCH_KEY = "epoch"
        self.BATCH_EPOCH_KEY = "batch/epoch"

        self.TEST_ERROR_KEY = "epoch/top1_error/test"
        self.BEST_TEST_ERROR_KEY = "epoch/top1_error_best/test"

        self.BATCH_KEY = "batch"
        self.ANGLE_KEYS = {"qy": "batch/angle_QY/global",
                           "fa": "batch/angle_fa/global_hidden",
                           "bp": "batch/angle_bp/global_hidden"
                           # "bp": "batch/angle_bp/global_hidden_average"
                           }

        self.APICAL_MAGNITUDE_KEY = "batch/apical_magnitude/global"

    def get_group_params(self, group):
        GROUP_PARAM_SETS = {
            "performance": dict(
                model_types=["ann", "burstccn"],
                modes=["fa", "kp", "tied"]
            )
        }
        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "performance":
            model_type = kwargs['model_type']
            mode = kwargs['mode']
            run_name = f"cifar10_{model_type}_{mode}_relu_rms"
        else:
            raise ValueError(f"Invalid group: {group}")

        return run_name

    def get_wandb_group_name(self, group, **kwargs):
        group_dict = {
            'performance': 'cifar_runs',
        }

        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        wandb_run_name = self.get_wandb_run_name(group, **kwargs)
        wandb_group = self.get_wandb_group_name(group, **kwargs)

        return {"run_name": wandb_run_name,
                "group": wandb_group}


if __name__ == "__main__":
    res = CIFARResultsStore()

    group = 'performance'
    run_params = res.get_group_params(group)

    modes = run_params['modes']
    model_types = run_params['model_types']

    wandb_run_filter = res.get_wandb_run_filter(group, model_type=model_types[0], mode=modes[0])
    print(wandb_run_filter)
    results = res.fetch(**wandb_run_filter, keys='batch')
    print(len(results))
