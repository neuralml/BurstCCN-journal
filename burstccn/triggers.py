from collections import defaultdict
import json
from pathlib import Path

import torch.nn.functional as F

import h5py
from torch import nn

import random
import numpy as np
import torch
from functools import wraps

def preserve_rng_state(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save RNG states
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_cpu_state = torch.get_rng_state()
        torch_cuda_state = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        try:
            return func(*args, **kwargs)
        finally:
            # Restore RNG states
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_state)

    return wrapper



class Trigger:
    def __init__(self, **kwargs): pass
    def on_epoch_start(self, epoch, trainer): pass
    def on_epoch_end(self, epoch, trainer): pass
    def on_batch_start(self, epoch, batch_index, trainer, inputs, targets): pass
    def on_batch_pre_update(self, epoch, batch_index, trainer, inputs, targets): pass
    def on_batch_end(self, epoch, batch_index, trainer, inputs, targets): pass


class ZeroWeightsTrigger(Trigger):
    def __init__(self, trigger_epoch, layer_index, **kwargs):
        super().__init__(**kwargs)
        self.trigger_epoch = trigger_epoch
        self.layer_index = layer_index

    def on_epoch_start(self, epoch, trainer):
        if epoch == self.trigger_epoch:
            layer = trainer.model.get_layers()[self.layer_index]
            with torch.no_grad():
                layer.W_weight.zero_()
            print(f"[Trigger] Zeroed weights of layer {self.layer_index} at start of epoch {epoch}")


class SimpleIncreasingTaskTrigger(Trigger):
    def __init__(self, increase_at_epoch, input_value, new_target_value, **kwargs):
        super().__init__(**kwargs)
        self.increase_at_epoch = increase_at_epoch
        self.input_value = input_value
        self.new_target_value = new_target_value

    def on_epoch_start(self, epoch, trainer):
        if epoch == 0:
            trainer.train_loader.dataset.set_input_value(self.input_value)
        if epoch == self.increase_at_epoch:
            trainer.train_loader.dataset.set_target_value(self.new_target_value)
            trainer.test_loader.dataset.set_target_value(self.new_target_value)


class RelativeTargetChangeTrigger(Trigger):
    def __init__(self, target_change, target_set_epoch=0, clamp_min=0.0, clamp_max=1.0, **kwargs):
        super().__init__(**kwargs)
        self.target_change = float(target_change)
        self.target_set_epoch = int(target_set_epoch)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.initial_output_value = None
        self.new_target_value = None
        self._target_has_been_set = False

    def on_epoch_start(self, epoch, trainer):
        if self._target_has_been_set or epoch != self.target_set_epoch:
            return

        initial_output_value = self._measure_initial_output(trainer)
        new_target_value = min(self.clamp_max, max(self.clamp_min, initial_output_value + self.target_change))

        trainer.train_loader.dataset.set_target_value(new_target_value)
        trainer.test_loader.dataset.set_target_value(new_target_value)

        self.initial_output_value = initial_output_value
        self.new_target_value = new_target_value
        self._target_has_been_set = True

        print(f"[Trigger] Initial output = {self.initial_output_value:.6f}")
        print(f"[Trigger] target_change = {self.target_change:.6f}")
        print(f"[Trigger] New target value = {self.new_target_value:.6f}")

    def _measure_initial_output(self, trainer):
        model = trainer.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            inputs, targets = next(iter(trainer.test_loader))
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if trainer.task_type == "classification":
                target_vectors = F.one_hot(targets, num_classes=model.n_outputs).float()
            elif trainer.task_type == "regression":
                target_vectors = targets
            else:
                raise ValueError(f"Unsupported task_type: {trainer.task_type}")

            outputs = trainer.training_strategy.forward(model, inputs, target_vectors)
            initial_output_value = float(outputs.mean().item())

        if was_training:
            model.train()

        return initial_output_value


class WeightOverrideTrigger(Trigger):
    def __init__(self, trigger_epoch, weight_type, layer_index, value=None, **kwargs):
        super().__init__(**kwargs)
        self.trigger_epoch = int(trigger_epoch)
        self.weight_type = weight_type
        self.layer_index = int(layer_index)
        self.value = None if value is None else float(value)

    def on_epoch_start(self, epoch, trainer):
        if epoch != self.trigger_epoch or self.value is None:
            return

        layer = trainer.model.get_layers()[self.layer_index]
        parameter = getattr(layer, self.weight_type)
        with torch.no_grad():
            parameter.fill_(self.value)
        trainer.model.apply_weight_constraints()
        print(
            f"[Trigger] Overrode {self.weight_type} of layer {self.layer_index} "
            f"with value {self.value:.6f} at start of epoch {epoch}"
        )


class InterneuronAblationTrigger(Trigger):
    def __init__(self, ablation_epoch, interneuron_type, layer_index, **kwargs):
        super().__init__(**kwargs)
        self.ablation_epoch = ablation_epoch
        self.interneuron_type = interneuron_type
        self.layer_index = layer_index
        self.initial_p_t_epoch_offset = 0

        self.initial_W_weight = None
        self.final_W_weight = None
        self.initial_W_direct_weight = None
        self.final_W_direct_weight = None
        self.initial_e = None
        self.final_e = None
        self.initial_p_t = None
        self.final_p_t = None

    @staticmethod
    def _scalar_value(value):
        return float(np.asarray(value, dtype=float).mean())

    def on_epoch_start(self, epoch, trainer):
        if epoch == self.ablation_epoch:
            layer = trainer.model.get_layers()[self.layer_index]

            if self.interneuron_type == 'sst':
                layer.Y_from_SST1_weight *= 0.0
                layer.Y_from_SST2_weight *= 0.0
            elif self.interneuron_type == 'ndnf':
                layer.Q_from_NDNF_weight *= 0.0
            elif self.interneuron_type == 'vip':
                layer.Y_VIP1_to_SST1_weight *= 0.0
                layer.Y_VIP2_to_SST2_weight *= 0.0

                # layer.sst1_bias.copy_(-((layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) * (
                #         (layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) < 0)).sum(dim=0))
                # layer.sst2_bias.copy_(-((layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) * (
                #         (layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) < 0)).sum(dim=0))

            elif self.interneuron_type == 'pv':
                # layer.W_to_PV_weight *= 0.0
                layer.W_from_PV_weight *= 0.0
            elif self.interneuron_type == 'control':
                pass
            else:
                raise ValueError(f"Invalid interneuron type {self.interneuron_type}")

            trainer.model.apply_weight_constraints()

    def on_batch_pre_update(self, epoch, batch_index, trainer, inputs, targets):
        if epoch == self.ablation_epoch and batch_index == 0 and self.initial_W_weight is None:
            layer = trainer.model.get_layers()[self.layer_index]
            self.initial_W_weight = layer.W_weight.clone().cpu().numpy()
            self.initial_W_direct_weight = layer.W_direct_weight.clone().cpu().numpy()
            self.initial_e = layer.e.clone().cpu().numpy()
        if (
                epoch == self.ablation_epoch + self.initial_p_t_epoch_offset
                and batch_index == 0
                and self.initial_p_t is None
        ):
            layer = trainer.model.get_layers()[self.layer_index]
            initial_p_t = getattr(layer, "p_t", None)
            self.initial_p_t = initial_p_t.clone().cpu().numpy() if initial_p_t is not None else None
        if epoch >= self.ablation_epoch:
            layer = trainer.model.get_layers()[self.layer_index]
            if self.interneuron_type == 'sst':
                layer.Y_from_SST1_weight.grad = None
                layer.Y_from_SST2_weight.grad = None
            elif self.interneuron_type == 'ndnf':
                layer.Q_from_NDNF_weight.grad = None
            elif self.interneuron_type == 'vip':
                layer.Y_VIP1_to_SST1_weight.grad = None
                layer.Y_VIP2_to_SST2_weight.grad = None
            elif self.interneuron_type == 'pv':
                layer.W_from_PV_weight.grad = None

    def on_epoch_end(self, epoch, trainer):
        if epoch == self.ablation_epoch+149:
            layer = trainer.model.get_layers()[self.layer_index]
            self.final_W_weight = layer.W_weight.clone().cpu().numpy()
            self.final_W_direct_weight = layer.W_direct_weight.clone().cpu().numpy()
            self.final_e = layer.e.clone().cpu().numpy()
            final_p_t = getattr(layer, "p_t", None)
            self.final_p_t = final_p_t.clone().cpu().numpy() if final_p_t is not None else None

            print(f"[InterneuronAblationTrigger] ablation_type = {self.interneuron_type}")
            if self.initial_W_direct_weight is not None:
                print(
                    f"[InterneuronAblationTrigger] w_direct_initial = "
                    f"{self._scalar_value(self.initial_W_direct_weight):.8f}"
                )
            if self.final_W_direct_weight is not None:
                print(
                    f"[InterneuronAblationTrigger] w_direct_final = "
                    f"{self._scalar_value(self.final_W_direct_weight):.8f}"
                )
            if self.initial_W_direct_weight is not None and self.final_W_direct_weight is not None:
                print(
                    f"[InterneuronAblationTrigger] w_direct_diff = "
                    f"{self._scalar_value(self.final_W_direct_weight - self.initial_W_direct_weight):.8f}"
                )
            if self.initial_W_weight is not None:
                print(f"[InterneuronAblationTrigger] w_initial = {self._scalar_value(self.initial_W_weight):.8f}")
            if self.final_W_weight is not None:
                print(f"[InterneuronAblationTrigger] w_final = {self._scalar_value(self.final_W_weight):.8f}")
            if self.initial_W_weight is not None and self.final_W_weight is not None:
                print(
                    f"[InterneuronAblationTrigger] w_diff = "
                    f"{self._scalar_value(self.final_W_weight - self.initial_W_weight):.8f}"
                )
            if self.initial_e is not None:
                print(f"[InterneuronAblationTrigger] event_rate_initial = {self._scalar_value(self.initial_e):.8f}")
            if self.final_e is not None:
                print(f"[InterneuronAblationTrigger] event_rate_final = {self._scalar_value(self.final_e):.8f}")
            if self.initial_e is not None and self.final_e is not None:
                print(
                    f"[InterneuronAblationTrigger] event_rate_diff = "
                    f"{self._scalar_value(self.final_e - self.initial_e):.8f}"
                )
            if self.initial_p_t is not None:
                print(f"[InterneuronAblationTrigger] p_t_initial = {self._scalar_value(self.initial_p_t):.8f}")
            if self.final_p_t is not None:
                print(f"[InterneuronAblationTrigger] p_t_final = {self._scalar_value(self.final_p_t):.8f}")
            if self.initial_p_t is not None and self.final_p_t is not None:
                print(
                    f"[InterneuronAblationTrigger] p_t_diff = "
                    f"{self._scalar_value(self.final_p_t - self.initial_p_t):.8f}"
                )


class ActivityTracker(Trigger):
    def __init__(self, activity_types, end_epochs, record_without_teacher=True, **kwargs):
        super().__init__(**kwargs)
        self.end_epochs = end_epochs
        # self.activity_types = ['sst', 'vip']
        self.activity_types = activity_types
        self.record_without_teacher = record_without_teacher

        if 'select_examples' in kwargs:
            self.select_examples = kwargs['select_examples']
        else:
            self.select_examples = None

        unique_run_name = kwargs['unique_run_name']
        self.save_path = Path("saved_activities") / f"{unique_run_name}.h5"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # self.save_path = f"SST_positive_negative_tracker_activities_layer_epochs_{start_tracking_epoch}_{end_tracking_epoch}.h5"
        # self.save_path = Path("saved_activities") / "activities.h5"

    def on_epoch_end(self, epoch, trainer):
        if epoch in self.end_epochs:
            activity_teacher = self.get_activity(trainer, self.activity_types)
            self._save_flat(epoch, activity_teacher, with_teacher=True)

            if self.record_without_teacher:
                activity_no_teacher = self.get_activity(trainer, self.activity_types, use_teacher=False)
                self._save_flat(epoch, activity_no_teacher, with_teacher=False)

    def _save_flat(self, epoch, activity_dict, with_teacher=True):
        teacher_key = "with_teacher" if with_teacher else "no_teacher"
        with h5py.File(self.save_path, "a") as f:
            for key, activity in activity_dict.items():
                if isinstance(key, tuple):
                    layer_idx, activity_type = key
                    full_key = f"epoch_{epoch}_layer_{layer_idx}_{activity_type}_{teacher_key}"
                else:
                    activity_type = key
                    full_key = f"epoch_{epoch}_{activity_type}_{teacher_key}"

                if full_key in f:
                    del f[full_key]
                f.create_dataset(full_key, data=activity.cpu().numpy(), compression="gzip")

    @preserve_rng_state
    def get_activity(self, trainer, activity_types, layer_indices=None, use_teacher=True):
        model = trainer.model
        dataset = trainer.test_loader

        all_layers = model.get_layers()
        # Determine which layers to use
        if layer_indices is None:
            layers = all_layers
            layer_indices = list(range(len(all_layers)))
        else:
            layers = [all_layers[i] for i in layer_indices]

        activity_list_dict = defaultdict(list)
        with torch.no_grad():
            for inputs, targets in dataset:
                backward_modes = [m for m in ("bp", "fa") if f"delta_{m}" in activity_types] if use_teacher else []
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                if trainer.task_type == 'classification':
                    target_vectors = F.one_hot(targets, num_classes=model.n_outputs).float()
                elif trainer.task_type == 'regression':
                    target_vectors = targets

                model.layers[-2].random_flip_apical = True

                trainer.training_strategy.forward_backward(
                    model, inputs, targets, target_vectors,
                    trainer.calc_loss, forward_noise=None,
                    use_teacher=use_teacher, backward_modes=backward_modes
                )

                model.layers[-2].random_flip_apical = False

                # ---- GLOBAL STATES (no layer_idx in key) ----
                if "inputs" in activity_types:
                    x = inputs
                    if self.select_examples is not None and x is not None:
                        x = x[:self.select_examples]
                    activity_list_dict["inputs"].append(x)

                if "targets" in activity_types:
                    y = targets.reshape(-1, 1)
                    if self.select_examples is not None and y is not None:
                        y = y[:self.select_examples]
                    activity_list_dict["targets"].append(y)

                # ---- LAYER STATES (layer_idx in key) ----
                for layer, layer_index in zip(layers, layer_indices):
                    for activity_type in activity_types:
                        if activity_type in ("inputs", "targets"):
                            continue  # already handled globally

                        activity = layer.get_state(activity_type)

                        if self.select_examples is not None and activity is not None:
                            activity = activity[:self.select_examples]

                        if activity is not None:
                            activity_list_dict[(layer_index, activity_type)].append(activity)

        activity_dict = {}
        for key, activity_list in activity_list_dict.items():
            activity_dict[key] = torch.cat(activity_list, dim=0)

        return activity_dict

class RankTracker(Trigger):
    def __init__(
        self,
        eval_epochs,
        state_key="apic",                 # your FA / approximate feedback signal
        bp_state_key="b_input_bp",        # BP signal before the local activation derivative
        max_examples=None,
        use_teacher=True,
        eps=1e-12,
        bp_var_threshold=0.95,            # choose BP subspace dimension by explained variance
        bp_topk=None,                     # optional fixed k instead of variance threshold
        subspace_compare_topk=None,       # optional k for FA-vs-BP principal angle comparison
        move_to_cpu=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_epochs = set(eval_epochs)
        self.state_key = state_key
        self.bp_state_key = bp_state_key
        self.max_examples = int(max_examples) if max_examples is not None else None
        self.use_teacher = bool(use_teacher)
        self.eps = float(eps)
        self.bp_var_threshold = float(bp_var_threshold)
        self.bp_topk = None if bp_topk is None else int(bp_topk)
        self.subspace_compare_topk = (
            None if subspace_compare_topk is None else int(subspace_compare_topk)
        )
        self.move_to_cpu = bool(move_to_cpu)

    def on_epoch_end(self, epoch, trainer):
        if epoch not in self.eval_epochs:
            return
        self.evaluate_and_print(epoch, trainer)

    @preserve_rng_state
    def evaluate_and_print(self, epoch, trainer):
        model = trainer.model
        layers = model.get_layers()
        was_training = model.training
        model.eval()

        fa_states = defaultdict(list)
        bp_states = defaultdict(list)
        seen = 0

        with torch.no_grad():
            for inputs, targets in trainer.test_loader:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                if trainer.task_type == "classification":
                    target_vectors = F.one_hot(
                        targets, num_classes=model.n_outputs
                    ).float()
                elif trainer.task_type == "regression":
                    target_vectors = targets
                else:
                    raise ValueError(f"Unsupported task_type: {trainer.task_type}")

                batch_size = inputs.shape[0]
                take = batch_size
                if self.max_examples is not None:
                    remaining = self.max_examples - seen
                    if remaining <= 0:
                        break
                    take = min(take, remaining)

                inputs_b = inputs[:take]
                targets_b = targets[:take]
                target_vectors_b = target_vectors[:take]
                backward_modes = []
                if self.bp_state_key in ("delta_bp", "b_input_bp"):
                    backward_modes.append("bp")

                trainer.training_strategy.clean_up(model, trainer.optimiser)
                trainer.training_strategy.forward_backward(
                    model,
                    inputs_b,
                    targets_b,
                    target_vectors_b,
                    trainer.calc_loss,
                    forward_noise=None,
                    use_teacher=self.use_teacher,
                    backward_modes=backward_modes,
                )

                for layer_idx, layer in enumerate(layers):
                    fa_state = self._get_layer_state(layer, self.state_key)
                    bp_state = self._get_bp_state(layer)

                    if fa_state is not None:
                        fa_states[layer_idx].append(
                            self._prepare_state_matrix(fa_state[:take])
                        )
                    if bp_state is not None:
                        bp_states[layer_idx].append(
                            self._prepare_state_matrix(bp_state[:take])
                        )

                seen += take
                if self.max_examples is not None and seen >= self.max_examples:
                    break

        print("\n" + "=" * 140)
        print(
            f"[RankTracker] epoch={epoch} examples={seen} "
            f"use_teacher={self.use_teacher} "
            f"fa_state={self.state_key} bp_state={self.bp_state_key}"
        )
        print("-" * 140)

        for layer_idx, layer in enumerate(layers):
            fa_chunks = fa_states.get(layer_idx, [])
            bp_chunks = bp_states.get(layer_idx, [])
            layer_name = getattr(layer, "name", None)
            tag = f"Layer {layer_idx:02d}" + (f" ({layer_name})" if layer_name else "")

            if len(fa_chunks) == 0:
                print(f"{tag}: missing FA state '{self.state_key}'")
                continue
            if len(bp_chunks) == 0:
                print(f"{tag}: missing BP state '{self.bp_state_key}'")
                continue

            X_fa = torch.cat(fa_chunks, dim=0)
            X_bp = torch.cat(bp_chunks, dim=0)

            n = min(X_fa.shape[0], X_bp.shape[0])
            X_fa = X_fa[:n]
            X_bp = X_bp[:n]

            metrics = self._compute_all_metrics(X_fa, X_bp)

            print(
                f"{tag}: "
                f"fa_pr={metrics['fa_pr']:.3f} "
                f"fa_erank={metrics['fa_erank']:.3f} | "
                f"bp_pr={metrics['bp_pr']:.3f} "
                f"bp_erank={metrics['bp_erank']:.3f} | "
                f"bp_k={metrics['bp_k']} "
                f"task_energy_ratio={metrics['task_energy_ratio']:.4f} "
                f"task_pr={metrics['task_pr']:.3f} "
                f"task_erank={metrics['task_erank']:.3f} | "
                f"fa_bp_cosine={metrics['fa_bp_cosine']:.4f} "
                f"subspace_overlap={metrics['subspace_overlap']:.4f} "
                f"mean_angle={metrics['mean_principal_angle_deg']:.2f}deg "
                f"max_angle={metrics['max_principal_angle_deg']:.2f}deg "
                f"(examples={n}, fa_features={X_fa.shape[1]}, bp_features={X_bp.shape[1]})"
            )
            print(
                "[RankTrackerResult] "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "layer": layer_idx,
                        "examples": n,
                        "fa_features": X_fa.shape[1],
                        "bp_features": X_bp.shape[1],
                        **metrics,
                    },
                    sort_keys=True,
                )
            )

        print("=" * 140 + "\n")

        if was_training:
            model.train()

    def _get_layer_state(self, layer, key):
        if hasattr(layer, "get_state"):
            state = layer.get_state(key)
            if isinstance(state, torch.Tensor):
                return state
        return None

    def _get_bp_state(self, layer):
        if hasattr(layer, self.bp_state_key):
            state = getattr(layer, self.bp_state_key)
            if isinstance(state, torch.Tensor):
                return state
        return self._get_layer_state(layer, self.bp_state_key)

    def _prepare_state_matrix(self, state):
        state = self._reshape_batch_feature_matrix(state).detach()
        state = state.float()
        if self.move_to_cpu:
            state = state.cpu()
        return state

    @staticmethod
    def _reshape_batch_feature_matrix(state):
        if state.ndim == 1:
            state = state.unsqueeze(1)
        elif state.ndim > 2:
            state = state.flatten(start_dim=1)
        return state

    def _center(self, X):
        X = X.float()
        if X.numel() == 0:
            return X
        return X - X.mean(dim=0, keepdim=True)

    def _safe_svdvals(self, X):
        if X.numel() == 0 or min(X.shape) == 0:
            return torch.zeros(0, device=X.device, dtype=X.dtype)
        return torch.linalg.svdvals(X)

    def _rank_metrics_from_centered(self, Xc):
        """
        Returns several notions of dimensionality on centered data matrix Xc
        of shape [examples, features].
        """
        out = {
            "algebraic_rank": 0.0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "effective_rank": 0.0,
            "energy": 0.0,
        }

        if Xc.numel() == 0 or min(Xc.shape) == 0:
            return out
        if not torch.isfinite(Xc).all():
            for k in out:
                out[k] = float("nan")
            return out

        s = self._safe_svdvals(Xc)
        if s.numel() == 0:
            return out

        s2 = s.pow(2)
        total = s2.sum()
        if total <= self.eps:
            return out

        smax2 = s2.max()
        tol = max(Xc.shape) * torch.finfo(s.dtype).eps * s.max()
        algebraic_rank = (s > tol).sum().item()

        p = s2 / (total + self.eps)
        p = p[p > 0]
        entropy = -(p * torch.log(p + self.eps)).sum()
        effective_rank = torch.exp(entropy)

        stable_rank = total / (smax2 + self.eps)
        participation_ratio = (total ** 2) / (s2.pow(2).sum() + self.eps)

        out["algebraic_rank"] = float(algebraic_rank)
        out["stable_rank"] = float(stable_rank.item())
        out["participation_ratio"] = float(participation_ratio.item())
        out["effective_rank"] = float(effective_rank.item())
        out["energy"] = float(total.item())
        return out

    def _compute_rank_metrics(self, X):
        Xc = self._center(X)
        return self._rank_metrics_from_centered(Xc)

    def _top_feature_basis(self, Xc, topk=None, var_threshold=0.95):
        """
        Returns V_k with shape [features, k], where columns are top right singular vectors
        of centered Xc.
        """
        if Xc.numel() == 0 or min(Xc.shape) == 0:
            return None, 0, None

        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        s2 = S.pow(2)
        total = s2.sum()

        if total <= self.eps or Vh.numel() == 0:
            return None, 0, None

        if topk is None:
            cumvar = torch.cumsum(s2, dim=0) / (total + self.eps)
            k = int(torch.searchsorted(cumvar, torch.tensor(var_threshold, device=cumvar.device)).item()) + 1
        else:
            k = int(topk)

        k = max(1, min(k, Vh.shape[0]))
        Vk = Vh[:k].transpose(0, 1).contiguous()   # [features, k]
        return Vk, k, S

    def _project_onto_basis(self, Xc, Vk):
        """
        Xc: [examples, features]
        Vk: [features, k]
        returns coefficients [examples, k] and projected reconstruction [examples, features]
        """
        if Vk is None:
            coeffs = Xc.new_zeros((Xc.shape[0], 0))
            recon = Xc.new_zeros(Xc.shape)
            return coeffs, recon
        coeffs = Xc @ Vk
        recon = coeffs @ Vk.transpose(0, 1)
        return coeffs, recon

    def _flat_cosine(self, A, B):
        a = A.reshape(-1)
        b = B.reshape(-1)
        na = torch.norm(a)
        nb = torch.norm(b)
        if na <= self.eps or nb <= self.eps:
            return 0.0
        return float((torch.dot(a, b) / (na * nb + self.eps)).item())

    def _principal_angle_metrics(self, X_fa_c, X_bp_c):
        """
        Compare top FA and BP feature subspaces.
        """
        fa_k = self.subspace_compare_topk
        bp_k = self.subspace_compare_topk

        Vfa, k_fa, _ = self._top_feature_basis(
            X_fa_c,
            topk=fa_k,
            var_threshold=self.bp_var_threshold,
        )
        Vbp, k_bp, _ = self._top_feature_basis(
            X_bp_c,
            topk=bp_k,
            var_threshold=self.bp_var_threshold,
        )

        if Vfa is None or Vbp is None or k_fa == 0 or k_bp == 0:
            return {
                "subspace_overlap": 0.0,
                "mean_principal_angle_deg": 90.0,
                "max_principal_angle_deg": 90.0,
            }

        k = min(Vfa.shape[1], Vbp.shape[1])
        Vfa = Vfa[:, :k]
        Vbp = Vbp[:, :k]

        M = Vfa.transpose(0, 1) @ Vbp
        s = torch.linalg.svdvals(M)
        s = torch.clamp(s, 0.0, 1.0)

        overlap = (s.pow(2).mean()).item()
        angles = torch.rad2deg(torch.arccos(s))
        mean_angle = angles.mean().item()
        max_angle = angles.max().item()

        return {
            "subspace_overlap": float(overlap),
            "mean_principal_angle_deg": float(mean_angle),
            "max_principal_angle_deg": float(max_angle),
        }

    def _compute_all_metrics(self, X_fa, X_bp):
        """
        X_fa and X_bp are [examples, features].
        """
        X_fa_c = self._center(X_fa)
        X_bp_c = self._center(X_bp)

        fa_metrics = self._rank_metrics_from_centered(X_fa_c)
        bp_metrics = self._rank_metrics_from_centered(X_bp_c)

        # BP task subspace
        Vbp, bp_k, _ = self._top_feature_basis(
            X_bp_c,
            topk=self.bp_topk,
            var_threshold=self.bp_var_threshold,
        )

        if Vbp is None or bp_k == 0:
            task_energy_ratio = 0.0
            task_proj_metrics = {
                "participation_ratio": 0.0,
                "effective_rank": 0.0,
            }
        else:
            coeffs_fa_in_bp, X_fa_task = self._project_onto_basis(X_fa_c, Vbp)

            fa_energy = (X_fa_c.pow(2).sum()).item()
            task_energy = (X_fa_task.pow(2).sum()).item()
            task_energy_ratio = float(task_energy / (fa_energy + self.eps))

            # rank of the FA signal restricted to the BP/task subspace
            task_proj_metrics = self._rank_metrics_from_centered(coeffs_fa_in_bp)

        alignment = self._flat_cosine(X_fa_c, X_bp_c)
        angle_metrics = self._principal_angle_metrics(X_fa_c, X_bp_c)

        return {
            "fa_pr": fa_metrics["participation_ratio"],
            "fa_erank": fa_metrics["effective_rank"],
            "fa_stable_rank": fa_metrics["stable_rank"],
            "fa_alg_rank": fa_metrics["algebraic_rank"],

            "bp_pr": bp_metrics["participation_ratio"],
            "bp_erank": bp_metrics["effective_rank"],
            "bp_stable_rank": bp_metrics["stable_rank"],
            "bp_alg_rank": bp_metrics["algebraic_rank"],

            "bp_k": int(bp_k),
            "task_energy_ratio": float(task_energy_ratio),
            "task_pr": task_proj_metrics["participation_ratio"],
            "task_erank": task_proj_metrics["effective_rank"],

            "fa_bp_cosine": float(alignment),
            "subspace_overlap": angle_metrics["subspace_overlap"],
            "mean_principal_angle_deg": angle_metrics["mean_principal_angle_deg"],
            "max_principal_angle_deg": angle_metrics["max_principal_angle_deg"],
        }

class BurstProbPCATracker(Trigger):
    def __init__(
        self,
        end_epochs,
        burst_prob_state="p_t",
        max_components=None,
        variance_thresholds=(0.8, 0.9, 0.95, 0.99),
        save_raw_burst_probs=True,
        save_plots=True,
        plot_dpi=150,
        class_conditioned_num_classes=10,
        min_examples_per_class=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.end_epochs = set(end_epochs)
        self.burst_prob_state = burst_prob_state
        self.max_components = max_components
        self.variance_thresholds = tuple(float(x) for x in variance_thresholds)
        self.save_raw_burst_probs = save_raw_burst_probs
        self.save_plots = bool(save_plots)
        self.plot_dpi = int(plot_dpi)
        self.class_conditioned_num_classes = int(class_conditioned_num_classes)
        self.min_examples_per_class = int(min_examples_per_class)
        self._plot_backend_warned = False

        unique_run_name = kwargs["unique_run_name"]
        self.save_path = Path("saved_activities") / f"{unique_run_name}_burst_prob_pca.h5"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.plot_dir = Path("saved_activities") / f"{unique_run_name}_burst_prob_pca_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, trainer):
        if epoch not in self.end_epochs:
            return

        layer_burst_probs, targets = self._collect_burst_probs(trainer)
        self._save_epoch_stats(epoch, layer_burst_probs, targets, trainer.task_type)

    @preserve_rng_state
    def _collect_burst_probs(self, trainer):
        model = trainer.model
        dataset = trainer.test_loader
        layers = model.get_layers()

        activity_list_dict = defaultdict(list)
        target_list = []
        with torch.no_grad():
            for inputs, targets in dataset:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                target_list.append(targets.detach().cpu())

                if trainer.task_type == "classification":
                    target_vectors = F.one_hot(targets, num_classes=model.n_outputs).float()
                elif trainer.task_type == "regression":
                    target_vectors = targets
                else:
                    raise ValueError(f"Unsupported task_type: {trainer.task_type}")

                trainer.training_strategy.forward_backward(
                    model,
                    inputs,
                    targets,
                    target_vectors,
                    trainer.calc_loss,
                    forward_noise=None,
                    use_teacher=True,
                    backward_modes=[],
                )

                for layer_idx, layer in enumerate(layers):
                    burst_prob = layer.get_state(self.burst_prob_state)
                    if burst_prob is None:
                        continue
                    burst_prob = self._reshape_batch_feature_matrix(burst_prob)
                    activity_list_dict[layer_idx].append(burst_prob)

        layer_burst_probs = {
            layer_idx: torch.cat(activity_chunks, dim=0)
            for layer_idx, activity_chunks in activity_list_dict.items()
            if len(activity_chunks) > 0
        }
        all_targets = torch.cat(target_list, dim=0) if len(target_list) > 0 else torch.empty(0, dtype=torch.long)
        return layer_burst_probs, all_targets

    @staticmethod
    def _reshape_batch_feature_matrix(state):
        if state.ndim == 1:
            state = state.unsqueeze(1)
        elif state.ndim > 2:
            state = state.flatten(start_dim=1)
        return state

    @staticmethod
    def _compute_effective_rank_participation_ratio(explained_variance):
        total_var = explained_variance.sum()
        if total_var <= 0:
            return 0.0
        return float((total_var ** 2 / (explained_variance.pow(2).sum() + 1e-12)).item())

    def _compute_pca_stats(self, X):
        X = X.float()
        n_examples, n_features = X.shape
        if n_examples == 0 or n_features == 0:
            raise ValueError("Cannot compute PCA on empty burst probability matrix.")

        max_rank = min(n_examples, n_features)
        if self.max_components is not None:
            k = min(int(self.max_components), max_rank)
        else:
            k = max_rank

        # Center across examples before PCA.
        Xc = X - X.mean(dim=0, keepdim=True)
        U, S, _ = torch.pca_lowrank(Xc, q=k, center=False)
        del U

        denom = max(n_examples - 1, 1)
        explained_variance = (S ** 2) / denom
        if n_examples < 2:
            total_variance = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        else:
            total_variance = Xc.var(dim=0, unbiased=True).sum()

        if (not torch.isfinite(total_variance)) or total_variance <= 0:
            explained_variance_ratio = torch.zeros_like(explained_variance)
        else:
            explained_variance_ratio = explained_variance / (total_variance + 1e-12)

        cumulative = torch.cumsum(explained_variance_ratio, dim=0)

        n_components_for_thresholds = {}
        for thr in self.variance_thresholds:
            idx = torch.nonzero(cumulative >= thr, as_tuple=False)
            n_components = int(idx[0, 0].item() + 1) if idx.numel() > 0 else int(cumulative.numel())
            n_components_for_thresholds[thr] = n_components

        eff_rank_pr = self._compute_effective_rank_participation_ratio(explained_variance)

        return {
            "explained_variance_ratio": explained_variance_ratio.cpu(),
            "cumulative_explained_variance_ratio": cumulative.cpu(),
            "explained_variance": explained_variance.cpu(),
            "singular_values": S.cpu(),
            "n_components_for_thresholds": n_components_for_thresholds,
            "effective_rank_participation_ratio": eff_rank_pr,
        }

    def _save_epoch_stats(self, epoch, layer_burst_probs, targets, task_type):
        teacher_key = "with_teacher"
        state_key = self.burst_prob_state
        with h5py.File(self.save_path, "a") as f:
            for layer_idx, X in layer_burst_probs.items():
                mixed_stats = self._compute_pca_stats(X)

                if self.save_raw_burst_probs:
                    raw_key = f"epoch_{epoch}_layer_{layer_idx}_{state_key}_{teacher_key}_mixed"
                    if raw_key in f:
                        del f[raw_key]
                    f.create_dataset(raw_key, data=X.cpu().numpy(), compression="gzip")

                mixed_prefix = f"epoch_{epoch}_layer_{layer_idx}_{state_key}_{teacher_key}_mixed"
                self._write_pca_stats(f, mixed_prefix, mixed_stats)

                class_stats_by_class = {}
                if task_type == "classification" and targets.numel() > 0:
                    targets_device = targets.to(device=X.device)
                    for class_idx in range(self.class_conditioned_num_classes):
                        class_mask = targets_device == class_idx
                        class_count = int(class_mask.sum().item())
                        if class_count < self.min_examples_per_class:
                            continue

                        class_X = X[class_mask]
                        class_stats = self._compute_pca_stats(class_X)
                        class_stats_by_class[class_idx] = class_stats

                        if self.save_raw_burst_probs:
                            class_raw_key = f"epoch_{epoch}_layer_{layer_idx}_{state_key}_{teacher_key}_class_{class_idx}"
                            if class_raw_key in f:
                                del f[class_raw_key]
                            f.create_dataset(class_raw_key, data=class_X.cpu().numpy(), compression="gzip")

                        class_prefix = f"epoch_{epoch}_layer_{layer_idx}_{state_key}_{teacher_key}_class_{class_idx}"
                        self._write_pca_stats(f, class_prefix, class_stats)

                if self.save_plots:
                    self._save_layer_subplot_plot(
                        epoch=epoch,
                        layer_idx=layer_idx,
                        state_key=state_key,
                        teacher_key=teacher_key,
                        mixed_stats=mixed_stats,
                        class_stats_by_class=class_stats_by_class,
                    )

    def _write_pca_stats(self, h5_file, key_prefix, pca_stats):
        for suffix, value in (
            ("explained_variance_ratio", pca_stats["explained_variance_ratio"]),
            ("cumulative_explained_variance_ratio", pca_stats["cumulative_explained_variance_ratio"]),
            ("explained_variance", pca_stats["explained_variance"]),
            ("singular_values", pca_stats["singular_values"]),
        ):
            key = f"{key_prefix}_{suffix}"
            if key in h5_file:
                del h5_file[key]
            h5_file.create_dataset(key, data=value.numpy(), compression="gzip")

        for thr, n_components in pca_stats["n_components_for_thresholds"].items():
            thr_tag = str(thr).replace(".", "p")
            key = f"{key_prefix}_n_components_for_{thr_tag}"
            if key in h5_file:
                del h5_file[key]
            h5_file.create_dataset(key, data=np.array(n_components, dtype=np.int32))

        eff_rank_key = f"{key_prefix}_effective_rank_pr"
        if eff_rank_key in h5_file:
            del h5_file[eff_rank_key]
        h5_file.create_dataset(eff_rank_key, data=np.array(pca_stats["effective_rank_participation_ratio"]))

    def _save_layer_subplot_plot(self, epoch, layer_idx, state_key, teacher_key, mixed_stats, class_stats_by_class):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            if not self._plot_backend_warned:
                print(f"[BurstProbPCATracker] Plotting skipped (matplotlib unavailable): {exc}")
                self._plot_backend_warned = True
            return

        n_panels = self.class_conditioned_num_classes + 1  # mixed + one per class
        n_cols = 4
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows))
        axes = np.atleast_1d(axes).reshape(-1)

        def draw_panel(ax, stats, title):
            cumulative = stats["cumulative_explained_variance_ratio"].numpy()
            x = np.arange(1, cumulative.shape[0] + 1)
            eff_rank = stats["effective_rank_participation_ratio"]

            ax.plot(x, cumulative, color="C0", linewidth=1.8)
            ax.set_ylim(0.0, 1.01)
            ax.set_xlim(1, max(1, cumulative.shape[0]))
            for thr in self.variance_thresholds:
                n_components = stats["n_components_for_thresholds"][thr]
                ax.axhline(thr, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
                ax.axvline(n_components, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
            ax.set_title(f"{title}\nEffRank={eff_rank:.2f}", fontsize=9)
            ax.set_xlabel("Components")
            ax.set_ylabel("Cum. Var")

        draw_panel(axes[0], mixed_stats, "Mixed")

        for class_idx in range(self.class_conditioned_num_classes):
            ax = axes[class_idx + 1]
            if class_idx in class_stats_by_class:
                draw_panel(ax, class_stats_by_class[class_idx], f"Class {class_idx}")
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"Class {class_idx}\nInsufficient samples", ha="center", va="center", fontsize=8)

        for ax in axes[n_panels:]:
            ax.set_axis_off()

        fig.suptitle(
            f"Epoch {epoch} Layer {layer_idx} {state_key} ({teacher_key})\n"
            f"Mixed vs class-conditioned PCA",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])

        plot_path = self.plot_dir / f"epoch_{epoch}_layer_{layer_idx}_{state_key}_{teacher_key}_mixed_vs_class_pca.png"
        fig.savefig(plot_path, dpi=self.plot_dpi)
        plt.close(fig)


class WeightNoiseTrigger(Trigger):
    def __init__(self, trigger_epoch, weight_type, noise_std, **kwargs):
        super().__init__(**kwargs)
        self.trigger_epoch = trigger_epoch
        self.weight_type = weight_type #'Q_weight'
        self.noise_std = noise_std

    @preserve_rng_state
    def on_epoch_start(self, epoch, trainer):
        if epoch == self.trigger_epoch:
            with torch.no_grad():
                weight_parameters = trainer.model.get_parameters(self.weight_type)
                for layer_index, weights in enumerate(weight_parameters):
                    # noise = torch.zeros_like(weights)
                    # nn.init.normal_(noise, std=self.noise_std * weights.std())
                    # weights.add_(noise)
                    # print(f"[Trigger] Added noise to weights of layer {layer_index} at start of epoch {epoch}")
                    N = weights.numel()
                    k = max(1, int(0.001 * N))  # 0.01%

                    # randomly choose k unique indices
                    idx = torch.randperm(N)[:k]

                    # Example: flip their signs
                    with torch.no_grad():
                        weights.view(-1)[idx] *= -1


class WeightFreezeTrigger(Trigger):
    def __init__(self, trigger_epoch, weight_type, layer_index, **kwargs):
        super().__init__(**kwargs)
        self.trigger_epoch = trigger_epoch
        self.weight_type = weight_type #'Q_weight'
        self.layer_index = layer_index

        self.is_frozen = False

    def on_epoch_start(self, epoch, trainer):
        if epoch == self.trigger_epoch:
            self.is_frozen = True
            print(f"[Trigger] Frozen weights of layer {self.layer_index} at start of epoch {epoch}")

    def on_batch_pre_update(self, epoch, batch_index, trainer, inputs, targets):
            with torch.no_grad():
                weight_parameters = trainer.model.get_parameters(self.weight_type)
                for layer_index, weights in enumerate(weight_parameters):
                    if layer_index == self.layer_index:
                        if self.is_frozen:
                            # weights.grad.mul_(0.0)
                            if weights.grad is not None:
                                weights.grad = None


class BurstCCNFeedbackOnlyLearningTrigger(Trigger):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)

        self.block_pretrain_epochs = kwargs['block_pretrain_epochs']
        self.prevent_standard_feedback_update = kwargs['prevent_standard_feedback_update']
        self.forward_noise = kwargs['forward_noise']

        assert mode in ['every_batch', 'epoch_block_training', 'batch_block_training']
        self.mode = mode

        if self.mode == 'every_batch':
            self.updates_per_batch = kwargs['updates_per_batch']
            self.batch_frequency = kwargs['batch_frequency']

        if self.mode == 'epoch_block_training':
            self.block_training_epochs = kwargs['block_training_epochs']

        if self.mode == 'batch_block_training':
            self.block_training_batches = kwargs['block_training_batches']
            # self.batch_frequency = kwargs['batch_frequency']

    def on_epoch_end(self, epoch, trainer):
        if epoch == 0:
            for _ in range(self.block_pretrain_epochs):
                self.epoch_feedback_train(epoch, trainer)

        if self.mode == 'epoch_block_training':
            for _ in range(self.block_training_epochs):
                self.epoch_feedback_train(epoch, trainer)

        if self.mode == 'batch_block_training':
            self.block_feedback_train(self.block_training_batches, trainer)

    def on_batch_end(self, epoch, batch_index, trainer, inputs, targets):
        if self.mode == 'every_batch':
            if batch_index % self.batch_frequency == 0:
                for _ in range(self.updates_per_batch):
                    self.batch_feedback_train(epoch, batch_index, trainer, inputs, targets)

    def on_batch_pre_update(self, epoch, batch_index, trainer, inputs, targets):
        if self.prevent_standard_feedback_update:
            for name, param in trainer.model.named_parameters():
                if 'Y' in name:
                    if param.grad is not None:
                        param.grad = None

    def block_feedback_train(self, num_batches, trainer):
        print(f"Starting feedback_only training ({num_batches} batches)...")
        trainer.parallel_model.train()

        # Try to use trainer.epoch if it exists, otherwise default to 0
        epoch = getattr(trainer, 'epoch', 0)

        data_iter = iter(trainer.train_loader)

        for global_batch_index in range(num_batches):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                # Restart the loader when we run out of data
                data_iter = iter(trainer.train_loader)
                inputs, targets = next(data_iter)

            # Derive a per-epoch-like batch index if possible
            if hasattr(trainer.train_loader, '__len__'):
                batch_index = global_batch_index % len(trainer.train_loader)
            else:
                batch_index = global_batch_index

            self.batch_feedback_train(epoch, batch_index, trainer, inputs, targets)

    def epoch_feedback_train(self, epoch, trainer):
        print("Starting feedback_only training...")
        trainer.parallel_model.train()

        for batch_index, (inputs, targets) in enumerate(trainer.train_loader):
            self.batch_feedback_train(epoch, batch_index, trainer, inputs, targets)

    def batch_feedback_train(self, epoch, batch_index, trainer, inputs, targets):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if trainer.task_type == 'classification':
            target_vectors = F.one_hot(targets, num_classes=trainer.model.n_outputs).float()
        elif trainer.task_type == 'regression':
            target_vectors = targets

        trainer.training_strategy.clean_up(trainer.model, trainer.optimiser)
        outputs, loss = trainer.training_strategy.forward_backward(trainer.model, inputs, targets,
                                                                   target_vectors, trainer.calc_loss,
                                                                   # trainer.model.forward_noise,
                                                                   self.forward_noise,
                                                                   use_teacher=False,
                                                                   backward_modes=None)

        for name, param in trainer.model.named_parameters():
            if 'W' in name:
                if param.grad is not None:
                    param.grad = None
        #
        # for name, param in trainer.model.named_parameters():
        #     if 'Y' in name:
        #         if param.grad is not None:
        #             param.grad *= 5.0

        trainer.optimiser.step()
        trainer.model.apply_weight_constraints()


class WeightRealignmentTrigger(Trigger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_start(self, epoch, trainer):
        model = trainer.model
        layers = model.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            layer.Y_weight.copy_(next_layer.W_weight.detach().clone())
            layer.Q_weight.copy_(-next_layer.p_baseline * next_layer.W_weight.detach().clone())

        print(f"[Trigger] Realignment of Y and Q weights at start of epoch {epoch}")


class BottleneckVsFullFAAlignmentEvalTrigger(Trigger):
    """
    Compares the *learning* backward signal (default: bottlenecked) to the *full standard FA* signal
    (computed as an extra backward phase via backward_modes=['fa']).

    You specified:
      - default signals are ALWAYS computed:
          layer.get_state('delta') and layer.<W>.grad  (bottlenecked learning signal)
      - full FA signals are computed when backward_modes includes 'fa':
          layer.get_state('delta_fa') and layer.<W>.grad_fa

    This trigger runs on the test set at selected epochs and prints per-layer:
      - delta_cos_mean: mean over examples of cosine(delta, delta_fa)
      - delta_norm_ratio_mean: mean over examples of ||delta|| / ||delta_fa||
      - delta_energy_ratio: total ||delta||^2 / ||delta_fa||^2 over dataset
      - grad_cos: cosine(grad, grad_fa)
      - grad_norm_ratio: ||grad|| / ||grad_fa||

    Interpretation:
      - delta_cos_mean close to 1 => bottlenecked signal direction matches full FA per-example
      - grad_cos close to 1 => weight-update direction matches full FA
      - norm ratios indicate rescaling (important if your learning rate is fixed)
    """

    def __init__(
        self,
        eval_epochs,
        max_examples=20000,
        use_teacher=True,
        # default (learning) states/attrs
        delta_state="delta",
        weight_attr="W_weight",   # change to "W_direct_weight" if needed
        grad_attr="grad",         # default gradient attribute
        # full FA reference states/attrs (computed via backward_modes=['fa'])
        delta_ref_state="delta_fa",
        grad_ref_attr="grad_fa",
        # extra backward phases to request
        backward_modes=("fa",),
        eps=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_epochs = set(eval_epochs)
        self.max_examples = int(max_examples) if max_examples is not None else None
        self.use_teacher = use_teacher

        self.delta_state = delta_state
        self.delta_ref_state = delta_ref_state

        self.weight_attr = weight_attr
        self.grad_attr = grad_attr
        self.grad_ref_attr = grad_ref_attr

        self.backward_modes = list(backward_modes)
        self.eps = float(eps)

    def on_epoch_end(self, epoch, trainer):
        if epoch not in self.eval_epochs:
            return
        self.evaluate_and_print(epoch, trainer)

    @preserve_rng_state
    def evaluate_and_print(self, epoch, trainer):
        model = trainer.model
        layers = model.get_layers()
        model.eval()

        # Per-layer accumulators
        acc = defaultdict(lambda: {
            "n_ex": 0,
            "delta_cos_sum": 0.0,
            "delta_norm_ratio_sum": 0.0,
            "delta_norm2": 0.0,
            "delta_ref_norm2": 0.0,
            "grad_dot": 0.0,
            "grad_norm2": 0.0,
            "grad_ref_norm2": 0.0,
        })

        seen = 0

        with torch.no_grad():
            for inputs, targets in trainer.test_loader:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                if trainer.task_type == "classification":
                    target_vectors = F.one_hot(targets, num_classes=model.n_outputs).float()
                else:
                    target_vectors = targets

                bs = inputs.shape[0]
                take = bs
                if self.max_examples is not None:
                    remaining = self.max_examples - seen
                    if remaining <= 0:
                        break
                    take = min(take, remaining)

                inputs_b = inputs[:take]
                targets_b = targets[:take]
                target_vectors_b = target_vectors[:take]

                # One call: computes default (bottleneck) + extra ('fa') reference states/grads
                trainer.training_strategy.forward_backward(
                    model,
                    inputs_b,
                    targets_b,
                    target_vectors_b,
                    trainer.calc_loss,
                    forward_noise=None,
                    use_teacher=self.use_teacher,
                    backward_modes=self.backward_modes,
                )

                for li, layer in enumerate(layers):
                    d = self._get_state(layer, self.delta_state, take)
                    dr = self._get_state(layer, self.delta_ref_state, take)

                    if d is not None and dr is not None:
                        a = d.reshape(take, -1).float()
                        b = dr.reshape(take, -1).float()

                        a_norm = torch.linalg.norm(a, dim=1) + self.eps
                        b_norm = torch.linalg.norm(b, dim=1) + self.eps

                        cos = (a * b).sum(dim=1) / (a_norm * b_norm)
                        ratio = a_norm / b_norm  # ||delta|| / ||delta_fa||

                        acc[li]["delta_cos_sum"] += float(cos.sum().item())
                        acc[li]["delta_norm_ratio_sum"] += float(ratio.sum().item())
                        acc[li]["n_ex"] += int(take)

                        acc[li]["delta_norm2"] += float((a * a).sum().item())
                        acc[li]["delta_ref_norm2"] += float((b * b).sum().item())

                    g, gr = self._get_grads(layer)
                    if g is not None and gr is not None:
                        g = g.reshape(-1).float()
                        gr = gr.reshape(-1).float()
                        acc[li]["grad_dot"] += float((g * gr).sum().item())
                        acc[li]["grad_norm2"] += float((g * g).sum().item())
                        acc[li]["grad_ref_norm2"] += float((gr * gr).sum().item())

                seen += take
                if self.max_examples is not None and seen >= self.max_examples:
                    break

        # Print
        print("\n" + "=" * 90)
        print(f"[BN vs full FA] epoch={epoch}  examples={seen}  use_teacher={self.use_teacher}")
        print(f"  delta: {self.delta_state} (learning)  vs  {self.delta_ref_state} (full FA)")
        print(f"  grads: {self.weight_attr}.{self.grad_attr} (learning)  vs  {self.weight_attr}.{self.grad_ref_attr} (full FA)")
        print("-" * 90)

        for li, layer in enumerate(layers):
            a = acc[li]
            name = getattr(layer, "name", None)
            tag = f"Layer {li:02d}" + (f" ({name})" if name else "")

            if a["n_ex"] == 0:
                print(f"{tag}: (missing delta states)")
                continue

            delta_cos_mean = a["delta_cos_sum"] / (a["n_ex"] + self.eps)
            delta_norm_ratio_mean = a["delta_norm_ratio_sum"] / (a["n_ex"] + self.eps)
            delta_energy_ratio = a["delta_norm2"] / (a["delta_ref_norm2"] + self.eps)

            print(f"{tag}:")
            print(f"  delta_cos_mean={delta_cos_mean:.3f}   mean(||d||/||d_fa||)={delta_norm_ratio_mean:.3f}   "
                  f"total_energy_ratio={delta_energy_ratio:.3f}   (N={a['n_ex']})")

            if a["grad_norm2"] > 0 and a["grad_ref_norm2"] > 0:
                grad_cos = a["grad_dot"] / ((a["grad_norm2"] ** 0.5) * (a["grad_ref_norm2"] ** 0.5) + self.eps)
                grad_norm_ratio = (a["grad_norm2"] ** 0.5) / ((a["grad_ref_norm2"] ** 0.5) + self.eps)
                print(f"  grad_cos={grad_cos:.3f}   grad_norm_ratio={grad_norm_ratio:.3f}")
            else:
                print("  grad: (missing)")

        print("=" * 90 + "\n")

    def _get_state(self, layer, key, take):
        if not hasattr(layer, "get_state"):
            return None
        try:
            t = layer.get_state(key)
        except Exception:
            return None
        if t is None or not isinstance(t, torch.Tensor):
            return None

        t = t[:take]
        if t.ndim == 1:
            t = t.unsqueeze(1)
        elif t.ndim > 2:
            t = t.view(t.shape[0], -1)

        # If transposed (n_units, batch)
        if t.ndim == 2 and t.shape[0] != take and t.shape[1] == take:
            t = t.t()

        if t.shape[0] != take:
            return None
        return t

    def _get_grads(self, layer):
        if not hasattr(layer, self.weight_attr):
            return None, None
        W = getattr(layer, self.weight_attr)

        g = getattr(W, self.grad_attr, None)
        gr = getattr(W, self.grad_ref_attr, None)

        if not isinstance(g, torch.Tensor) or not isinstance(gr, torch.Tensor):
            return None, None
        return g.detach(), gr.detach()
