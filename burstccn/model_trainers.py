import os
import json
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from tqdm import tqdm

from burstccn.datasets.dataset_utils import get_dataloader_length
from burstccn.models.networks.base import AutogradNetwork
from burstccn.models.networks.np_base import NPBase
from burstccn.utils import topk_correct

def smooth_one_hot(targets, num_classes, smoothing: float = 0.1):
    """
    targets: (B,) long
    returns: (B, num_classes) float
    """
    assert 0.0 <= smoothing < 1.0
    with torch.no_grad():
        off_value = smoothing / (num_classes - 1)
        on_value = 1.0 - smoothing
        # (B, K)
        y = torch.full((targets.size(0), num_classes),
                       off_value,
                       device=targets.device,
                       dtype=torch.float32)
        y.scatter_(1, targets.unsqueeze(1), on_value)
    return y

class ModelTrainer:
    def __init__(self, model, parallel_model, optimiser, train_loader, val_loader, test_loader, task_type, loss_type, logger,
                 model_inspector, max_stagnant_epochs=None, save_models=False, model_output_dir='./', trigger_event_map=None,
                 label_smoothing=0.0):
        self.model = model
        self.parallel_model = parallel_model
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.logger = logger
        self.model_inspector = model_inspector
        self.max_stagnant_epochs = max_stagnant_epochs
        self.save_models = save_models
        self.model_output_dir = model_output_dir
        self.trigger_event_map = trigger_event_map

        self.label_smoothing = label_smoothing

        self.task_type = task_type
        if self.task_type == 'classification':
            self.metrics_tracker = ClassificationMetricsTracker()
        elif self.task_type == 'regression':
            self.metrics_tracker = RegressionMetricsTracker()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self.calc_loss = self._create_loss_function(loss_type)

        if isinstance(self.model, NPBase):
            self.training_strategy = NPTrainingStrategy()
        elif isinstance(self.model, AutogradNetwork):
            self.training_strategy = AutogradTrainingStrategy()
        else:
            self.training_strategy = NoAutogradTrainingStrategy()

        assert self.model.forward_noise is None or self.model.forward_noise > 0
        self.training_with_noise = self.model.forward_noise is not None and self.model.forward_noise > 0

        self.perform_noiseless_with_teacher_passes = self.training_with_noise or not self.model.use_teacher
        self.perform_noiseless_without_teacher_passes = self.training_with_noise and not self.model.use_teacher

        self.best_epoch_metrics = {}
        self.latest_epoch_metrics = {}

        self.is_slurm = "SLURM_JOB_ID" in os.environ
        print(f"Running on SLURM: {self.is_slurm}")

    def _create_loss_function(self, loss_type):
        if loss_type == 'mse':
            loss_fn = torch.nn.MSELoss().cuda()
            calc_loss = lambda outputs, targets, target_vectors: loss_fn(outputs, target_vectors)
        elif loss_type == 'cross_entropy':
            loss_fn = torch.nn.CrossEntropyLoss().cuda()
            calc_loss = lambda outputs, targets, target_vectors: loss_fn(outputs, targets)
        else:
            raise NotImplementedError
        return calc_loss

    def _last_layer_error_scale(self):
        return getattr(self.model.layers[-1], "error_scale", 1.0)

    def _teacher_effectively_used(self):
        return self.model.use_teacher and self._last_layer_error_scale() != 0.0

    @contextmanager
    def _temporary_last_layer_error_scale(self, value):
        last_layer = self.model.layers[-1]
        original_error_scale = getattr(last_layer, "error_scale", 1.0)
        try:
            last_layer.error_scale = value
            yield
        finally:
            last_layer.error_scale = original_error_scale

    def _run_triggers(self, event, **kwargs):
        for trig in self.trigger_event_map.get(event, []):
            getattr(trig, event)(**kwargs)

    def update_best_epoch_metrics(self, current_metrics, epoch, subset_name):
        for key, value in current_metrics.items():
            metric_name = key.split("/")[0]  # e.g. 'loss' from 'loss/train'
            if subset_name not in self.best_epoch_metrics:
                self.best_epoch_metrics[subset_name] = {}
            if metric_name not in self.best_epoch_metrics[subset_name] or value < self.best_epoch_metrics[subset_name][metric_name]["value"]:
                self.best_epoch_metrics[subset_name][metric_name] = {"value": value, "epoch": epoch}

    def update_latest_epoch_metrics(self, current_metrics, subset_name):
        self.latest_epoch_metrics[subset_name] = {}
        for key, value in current_metrics.items():
            metric_name = key.split("/")[0]
            self.latest_epoch_metrics[subset_name][metric_name] = value

    def update_and_log_epoch_metrics(self, subset_metrics_dict, epoch):
        # Build log data
        log_data = {}
        self.latest_epoch_metrics = {}

        # Log current metrics
        for subset_name, subset_metrics in subset_metrics_dict.items():
            self.update_latest_epoch_metrics(subset_metrics, subset_name)
            self.update_best_epoch_metrics(subset_metrics, epoch, subset_name)
            for key, value in subset_metrics.items():
                log_data[key] = value

        # Log best metrics
        for subset_name, metrics_dict in self.best_epoch_metrics.items():
            for metric_name, metric_info in metrics_dict.items():
                log_data[f"{metric_name}_best/{subset_name}"] = metric_info["value"]
                log_data[f"{metric_name}_best_epoch/{subset_name}"] = metric_info["epoch"]

        # Commit logs
        self.logger.log(log_data)
        self.logger.commit_log_epoch(epoch=epoch)

    def get_training_result(self):
        result = {}
        for split_name in ("train", "val", "test"):
            top1_error = self.latest_epoch_metrics.get(split_name, {}).get("top1_error")
            if top1_error is not None:
                result[f"epoch/top1_error/{split_name}"] = top1_error

            best_top1_error = self.best_epoch_metrics.get(split_name, {}).get("top1_error", {}).get("value")
            if best_top1_error is not None:
                result[f"epoch/top1_error_best/{split_name}"] = best_top1_error
        return result

    def print_training_result(self):
        print("[TrainingResult] " + json.dumps(self.get_training_result(), sort_keys=True))

    def train_with_evaluation(self, n_epochs):
        training_context = self.training_strategy.get_context()

        with training_context:
            self._run_triggers("on_epoch_start", epoch=0, trainer=self)
            val_metrics = self.evaluate(self.val_loader, subset_name='val', epoch=0) if self.val_loader else None
            test_metrics = self.evaluate(self.test_loader, subset_name='test', epoch=0)

            self._run_triggers("on_epoch_end", epoch=0, trainer=self)

            if self.save_models:
                # torch.save(self.model.state_dict(), os.path.join(self.model_output_dir, "best_top5_test_error_model.pth"))
                torch.save(self.model.state_dict(), os.path.join(self.model_output_dir, "initial_model.pth"))

            subset_metrics_dict = {"test": test_metrics}
            if val_metrics is not None: subset_metrics_dict["val"] = val_metrics

            self.update_and_log_epoch_metrics(subset_metrics_dict, epoch=0)
            if self.save_models:
                self.save_model(epoch=0)

            for epoch in range(1, n_epochs):
                self._run_triggers("on_epoch_start", epoch=epoch, trainer=self)

                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader, subset_name='val', epoch=epoch) if self.val_loader else None
                test_metrics = self.evaluate(self.test_loader, subset_name='test', epoch=epoch)

                self._run_triggers("on_epoch_end", epoch=epoch, trainer=self)

                # Check for NaN values and stop training if found
                for key, value in {**train_metrics, **test_metrics}.items():
                    if torch.isnan(torch.tensor(value)):
                        print(f"Stopping training due to NaN in metric '{key}' at epoch {epoch}")
                        self.print_training_result()
                        return

                subset_metrics_dict = {"train": train_metrics, "test": test_metrics}
                if val_metrics is not None: subset_metrics_dict["val"] = val_metrics
                self.update_and_log_epoch_metrics(subset_metrics_dict, epoch=epoch)

                if self.save_models:
                    self.save_model(epoch=epoch)

                # Stop training if max stagnant epochs is reached
                best_top1_test_error_epoch = self.best_epoch_metrics.get("test", {}).get("top1_error", {}).get("epoch", 0)
                if self.max_stagnant_epochs is not None and epoch > best_top1_test_error_epoch + self.max_stagnant_epochs:
                    print(f"Stopping training due to no improvement in Top1 Test Error for {self.max_stagnant_epochs} epochs.")
                    self.print_training_result()
                    return

        self.print_training_result()

    def save_model(self, epoch):
        """Save model if current epoch matches the best val error epochs."""
        os.makedirs(self.model_output_dir, exist_ok=True)

        best_top1_val_error_epoch = (
            self.best_epoch_metrics.get("val", {})
            .get("top1_error", {})
            .get("epoch", -1)
        )
        if best_top1_val_error_epoch == epoch:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.model_output_dir, "best_top1_val_error_model.pth"),
            )

        best_top5_val_error_epoch = (
            self.best_epoch_metrics.get("val", {})
            .get("top5_error", {})
            .get("epoch", -1)
        )
        if best_top5_val_error_epoch == epoch:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.model_output_dir, "best_top5_val_error_model.pth"),
            )

    def train_epoch(self, epoch):
        self.parallel_model.train()

        n_batches = get_dataloader_length(self.train_loader)
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", total=n_batches, mininterval=20.0 if self.is_slurm else 0.1)

        self.metrics_tracker.reset("train", track_ema=True)

        for batch_index, (inputs, targets) in enumerate(train_progress_bar):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            self._run_triggers("on_batch_start", epoch=epoch, batch_index=batch_index, trainer=self, inputs=inputs, targets=targets)

            if self.task_type == 'classification':
                # label_smoothing = 0.0 #getattr(self, "label_smoothing", 0.1)
                # raise NotImplementedError
                if self.label_smoothing > 0.0:
                    target_vectors = smooth_one_hot(targets,
                                                    num_classes=self.model.n_outputs,
                                                    smoothing=self.label_smoothing)
                else:
                    target_vectors = F.one_hot(targets, num_classes=self.model.n_outputs).float()
                # target_vectors = F.one_hot(targets, num_classes=self.model.n_outputs).float()
            elif self.task_type == 'regression':
                target_vectors = targets

            if self.logger.should_log_batch(batch_index):
                if self.perform_noiseless_with_teacher_passes:
                    self.training_strategy.clean_up(self.model, self.optimiser)

                    if self._last_layer_error_scale() == 0.0:
                        with self._temporary_last_layer_error_scale(0.5):
                            _ = self.training_strategy.forward_backward(self.model, inputs, targets,
                                                                        target_vectors, self.calc_loss,
                                                                        forward_noise=None,
                                                                        use_teacher=True,
                                                                        backward_modes=['bp', 'fa'])
                    else:
                        _ = self.training_strategy.forward_backward(self.model, inputs, targets,
                                                                    target_vectors, self.calc_loss,
                                                                    forward_noise=None,
                                                                    use_teacher=True,
                                                                    backward_modes=['bp', 'fa'])

                    internal_state = self.model_inspector.get_internal_model_state(with_teacher=True,
                                                                                   with_forward_noise=False)

                    self.logger.log(internal_state)

                if self.perform_noiseless_without_teacher_passes:
                    self.training_strategy.clean_up(self.model, self.optimiser)
                    _ = self.training_strategy.forward_backward(self.model, inputs, targets,
                                                                            target_vectors, self.calc_loss,
                                                                            forward_noise=None,
                                                                            use_teacher=False)

                    internal_state = self.model_inspector.get_internal_model_state(with_teacher=False,
                                                                                   with_forward_noise=False)
                    self.logger.log(internal_state)

            self.training_strategy.clean_up(self.model, self.optimiser)

            if self.logger.should_log_batch(batch_index) and not self.perform_noiseless_with_teacher_passes:
                backward_modes = ['bp', 'fa']
            else:
                backward_modes = None

            # backward_modes = None

            outputs, loss = self.training_strategy.forward_backward(self.model, inputs, targets,
                                                                    target_vectors, self.calc_loss,
                                                                    self.model.forward_noise,
                                                                    self.model.use_teacher,
                                                                    backward_modes=backward_modes)

            # #TODO TEMP:
            # for name, param in self.model.named_parameters():
            #     if not hasattr(param, 'grad_bp'):
            #         continue
            #     param.grad = param.grad_bp

            # if epoch <= 5:
            #     for name, p in self.model.named_parameters():
            #         if p.grad is not None and "bias" not in name.lower():
            #             p.grad += 5e-2 * p

            self._run_triggers("on_batch_pre_update", epoch=epoch, batch_index=batch_index, trainer=self, inputs=inputs, targets=targets)

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimiser.step()
            self.optimiser.step_schedule()
            self.model.apply_weight_constraints()

            if self.logger.should_log_batch(batch_index):
                teacher_used = self._teacher_effectively_used()
                internal_state = self.model_inspector.get_internal_model_state(with_teacher=teacher_used,
                                                                               with_forward_noise=self.training_with_noise)
                self.logger.log(internal_state)

            self.metrics_tracker.update(outputs, targets, loss)

            train_progress_bar.set_description(self.metrics_tracker.get_description(epoch), refresh=False)

            if self.logger.should_log_batch(batch_index):
                batch = (epoch - 1) * len(self.train_loader) + batch_index
                self.logger.commit_log_batch(batch=batch,
                                             batch_index=batch_index,
                                             epoch=epoch)

            self._run_triggers("on_batch_end", epoch=epoch, batch_index=batch_index, trainer=self, inputs=inputs, targets=targets)

        metrics = self.metrics_tracker.get_metrics()
        return metrics

    def evaluate(self, data_loader, subset_name, epoch):
        assert subset_name in ['train', 'val', 'test']
        self.model.eval()

        n_batches = get_dataloader_length(data_loader)
        eval_progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", total=n_batches, mininterval=20.0 if self.is_slurm else 0.1)
        self.metrics_tracker.reset(subset_name)

        with torch.no_grad():
            for inputs, targets in eval_progress_bar:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                if self.task_type == 'classification':
                    target_vectors = F.one_hot(targets, num_classes=self.model.n_outputs).float()
                elif self.task_type == 'regression':
                    target_vectors = targets

                outputs = self.training_strategy.forward(self.model, inputs, target_vectors)

                loss = self.calc_loss(outputs, targets, target_vectors)
                self.metrics_tracker.update(outputs, targets, loss.item())
                eval_progress_bar.set_description(self.metrics_tracker.get_description(epoch), refresh=False)

        metrics = self.metrics_tracker.get_metrics()
        return metrics


class ModelTrainingStrategy:
    def get_context(self):
        import contextlib
        return contextlib.nullcontext()

    def forward(self, model, inputs, target_vectors):
        """Return model outputs in evaluation mode."""
        raise NotImplementedError

    def forward_backward(self, model, inputs, targets, target_vectors, calc_loss_fn,
                         forward_noise, use_teacher, backward_modes=None):
        """Return outputs and scalar loss after applying backward logic."""
        raise NotImplementedError

    def clean_up(self, model, optimiser):
        optimiser.zero_grad()
        model.zero_grad_bp()
        model.zero_grad_fa()


class AutogradTrainingStrategy(ModelTrainingStrategy):
    def forward(self, model, inputs, target_vectors):
        return model.forward(inputs, target=target_vectors)

    def forward_backward(self, model, inputs, targets, target_vectors, calc_loss_fn,
                         forward_noise, use_teacher, backward_modes=None):
        if backward_modes is not None: raise NotImplementedError
        if not use_teacher: raise NotImplementedError
        if forward_noise is not None: raise NotImplementedError

        if backward_modes is None:
            backward_modes = []

        if "bp" in backward_modes:
            model.set_store_grad_bp(True)
            outputs = model.forward(inputs, target=target_vectors, use_backprop=True)
            loss = calc_loss_fn(outputs, targets, target_vectors)
            loss.backward()
            model.set_store_grad_bp(False)

        if "fa" in backward_modes:
            model.set_store_grad_fa(True)
            outputs = model.forward(inputs, target=target_vectors, use_feedback_alignment=True)
            loss = calc_loss_fn(outputs, targets, target_vectors)
            loss.backward()
            model.set_store_grad_fa(False)

        outputs = model.forward(inputs, target=target_vectors)
        loss = calc_loss_fn(outputs, targets, target_vectors)
        loss.backward()

        return outputs, loss.item()


class NoAutogradTrainingStrategy(ModelTrainingStrategy):
    def get_context(self):
        return torch.no_grad()

    def forward(self, model, inputs, target_vectors):
        return model.forward(inputs)

    def forward_backward(self, model, inputs, targets, target_vectors, calc_loss_fn,
                         forward_noise, use_teacher, backward_modes=None):
        if backward_modes is None:
            backward_modes = []

        if "bp" in backward_modes or "fa" in backward_modes:
            assert use_teacher

        outputs = model.forward(inputs, forward_noise=forward_noise)
        loss = calc_loss_fn(outputs, targets, target_vectors)

        if use_teacher:
            model.backward(target_vectors)
        else:
            model.backward(None)

        if "bp" in backward_modes: model.backward_bp(target_vectors)
        if "fa" in backward_modes: model.backward_fa(target_vectors)

        return outputs, loss.item()


class NPTrainingStrategy(ModelTrainingStrategy):
    def get_context(self):
        return torch.no_grad()

    def forward(self, model, inputs, target_vectors, perturb=False):
        return model.forward(inputs, perturb=perturb)

    def forward_backward(self, model, inputs, targets, target_vectors, calc_loss_fn,
                         forward_noise, use_teacher, backward_modes=None):
        if backward_modes is None:
            backward_modes = []

        if "bp" in backward_modes or "fa" in backward_modes:
            assert use_teacher

        loss_fn = torch.nn.MSELoss(reduction="none").cuda()

        outputs_pre = model.forward(inputs, forward_noise=forward_noise, perturb=False)
        loss_pre = loss_fn(outputs_pre, target_vectors).mean(dim=1)

        outputs_post = model.forward(inputs, forward_noise=forward_noise, perturb=True)
        loss_post = loss_fn(outputs_post, target_vectors).mean(dim=1)

        if use_teacher:
            model.backward(target_vectors, loss_pre=loss_pre, loss_post=loss_post)
        else:
            model.backward(None)

        if "bp" in backward_modes: model.backward_bp(target_vectors)
        if "fa" in backward_modes: model.backward_fa(target_vectors)

        loss_pre = calc_loss_fn(outputs_pre, targets, target_vectors)
        return outputs_pre, loss_pre.item()


class BaseMetricsTracker:
    def reset(self, subset_name):
        raise NotImplementedError

    def update(self, outputs, targets, loss):
        raise NotImplementedError

    def get_description(self, epoch):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError


class ClassificationMetricsTracker(BaseMetricsTracker):
    def reset(self, subset_name, track_ema=False):
        self.subset_name = subset_name
        self.track_ema = track_ema

        if self.track_ema:
            self.loss_ema = None
            self.loss_ema_alpha = 0.1

        self.total_loss = 0.0
        self.top1_correct = 0
        self.top5_correct = 0
        self.total_examples = 0

    def update(self, outputs, targets, loss):
        top1, top5 = topk_correct(outputs, targets, topk=(1, 5))
        self.top1_correct += top1
        self.top5_correct += top5
        num_examples = targets.size(0)
        self.total_examples += num_examples

        if self.track_ema:
            if self.loss_ema is None:
                self.loss_ema = loss
            else:
                self.loss_ema = self.loss_ema_alpha * loss + (1 - self.loss_ema_alpha) * self.loss_ema

        self.total_loss += loss * num_examples

    def get_description(self, epoch):
        if self.track_ema:
            loss_desc = "Loss (EMA): {:.3f}".format(self.loss_ema)
        else:
            avg_loss = self.total_loss / self.total_examples
            loss_desc = "Loss: {:.3f}".format(avg_loss)

        desc = "{:<5} | Epoch {:d} | {} | Top1 Acc: {:.3f}% ({:d}/{:d}) | Top5 Acc: {:.3f}% ({:d}/{:d})".format(
            self.subset_name.capitalize(),
            epoch,
            loss_desc,
            100 * self.top1_correct / self.total_examples, self.top1_correct, self.total_examples,
            100 * self.top5_correct / self.total_examples, self.top5_correct, self.total_examples)

        return desc

    def get_metrics(self):
        avg_loss = self.total_loss / self.total_examples
        top1_error = 100 - 100 * self.top1_correct / self.total_examples
        top5_error = 100 - 100 * self.top5_correct / self.total_examples
        return {
            f"loss/{self.subset_name}": avg_loss,
            f"top1_error/{self.subset_name}": top1_error,
            f"top5_error/{self.subset_name}": top5_error
        }


class RegressionMetricsTracker(BaseMetricsTracker):
    def reset(self, subset_name, track_ema=False):
        self.subset_name = subset_name
        self.track_ema = track_ema

        if self.track_ema:
            self.loss_ema = None
            self.loss_ema_alpha = 0.1

        self.total_loss = 0.0
        self.total_examples = 0

    def update(self, outputs, targets, loss):
        num_examples = targets.size(0)
        self.total_examples += num_examples

        if self.track_ema:
            if self.loss_ema is None:
                self.loss_ema = loss
            else:
                self.loss_ema = self.loss_ema_alpha * loss + (1 - self.loss_ema_alpha) * self.loss_ema

        self.total_loss += loss * num_examples

    def get_description(self, epoch):
        if self.track_ema:
            loss_desc = "Loss (EMA): {:.3f}".format(self.loss_ema)
        else:
            avg_loss = self.total_loss / self.total_examples
            loss_desc = "Loss: {:.3f}".format(avg_loss)

        desc = "{:<5} | Epoch {:d} | {} | Total Examples: {:d}".format(
            self.subset_name.capitalize(),
            epoch,
            loss_desc,
            self.total_examples
        )
        return desc

    def get_metrics(self):
        avg_loss = self.total_loss / self.total_examples
        return {
            f"loss/{self.subset_name}": avg_loss
        }
