from plotting.analysis.results_store_base import WandbResultsStore


class ImageTaskResultsStore(WandbResultsStore):
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
            "cifar10": dict(
                model_types=["ann", "burstccn"],
                modes=["fa", "kp", "tied"]
            ),
            "imagenet": dict(
                model_types=["ann", "burstccn"],
                modes=["fa", "kp", "tied"]
            )
        }

        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "cifar10":
            model_type = kwargs['model_type']
            mode = kwargs['mode']
            run_name = f"cifar10_{model_type}_{mode}_relu_rms"
        elif group == "imagenet":
            model_type = kwargs['model_type']
            mode = kwargs['mode']
            run_name = f"imagenet_{model_type}_{mode}_relu_rms"
        else:
            raise ValueError(f"Invalid group: {group}")

        return run_name

    def get_wandb_group_name(self, group, **kwargs):
        # if group == 'cifar10':
        #     if kwargs['model_type'] == 'ann':
        #         return 'cifar10_runs'
        #     else:
        #         return 'cifar10_feb1'

        group_dict = {
            # 'cifar10': 'cifar10_runs',
            # 'cifar10': 'cifar10_feb1',
            # 'cifar10': 'cifar10_feb2',
            'cifar10': 'cifar10_feb2_no_surrogate',
            # 'imagenet': 'imagenet_runs',
            # 'imagenet': 'imagenet_feb1',
            'imagenet': 'imagenet_feb1_no_surrogate',
        }

        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        # if group == "cifar10" and kwargs['mode'] == 'kp':
        #     model_type = kwargs['model_type']
        #     wandb_run_name = f"cifar10_{model_type}_kp_relu_rms_2"
        #     # wandb_run_name = f"cifar10_{model_type}_kp_relu_rms"
        #     wandb_group = self.get_wandb_group_name(group, **kwargs)
        # else:
        wandb_run_name = self.get_wandb_run_name(group, **kwargs)
        wandb_group = self.get_wandb_group_name(group, **kwargs)

        return {"run_name": wandb_run_name,
                "group": wandb_group}


if __name__ == "__main__":
    res = ImageTaskResultsStore()

    group = 'imagenet'
    run_params = res.get_group_params(group)

    modes = run_params['modes']
    model_types = run_params['model_types']

    wandb_run_filter = res.get_wandb_run_filter(group, model_type=model_types[0], mode=modes[0])
    print(wandb_run_filter)
    results = res.fetch(**wandb_run_filter, keys='batch')
    print(len(results))
