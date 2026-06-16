import wandb


def add_prefix(data, prefix):
    """
    Return a new dict with all keys prefixed for clear namespacing.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


class WandbLogger:
    def __init__(self, log_interval=200):
        self.log_interval = log_interval
        self.log_dict = dict()

        # Define custom x-axis metrics for your metric namespaces
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("batch/*", step_metric="batch")

    def should_log_batch(self, batch_index):
        return batch_index % self.log_interval == 0

    def log(self, data):
        self.log_dict.update(data)

    def commit_log_batch(self, batch, batch_index, epoch):
        prefixed_log = add_prefix(self.log_dict, "batch/")
        prefixed_log.update({
            "batch": batch,
            "batch/batch_index": batch_index,
            "batch/epoch": epoch
        })
        self._log_commit(prefixed_log)

    def commit_log_epoch(self, epoch):
        prefixed_log = add_prefix(self.log_dict, "epoch/")
        prefixed_log.update({
            "epoch": epoch
        })
        self._log_commit(prefixed_log)

    def _log_commit(self, log_data):
        wandb.log(log_data)
        self.log_dict = dict()
