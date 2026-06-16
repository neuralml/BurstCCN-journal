import time
import copy
import os
import shutil

import random
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

from continuous_burstccn import ContinuousBurstCCNNetwork
from continuous_burstccn_ma import MAContinuousBurstCCNNetwork
from datasets import CatCamContinuousDataLoader, ANN, SinWaveContinuousDataLoader


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContinuousModelTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        run_name,
        loss_fn=torch.nn.MSELoss(),
        device='cpu',
        dt=0.01,
        buffer_interval=1000,
        record_interval=100000,
        record_duration=10000,
        num_test_examples_before=500000,
        num_training_examples=10000000,
        num_test_examples_after=500000,
        base_output_dir="training_runs",
        overwrite=False
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.dt = dt

        self.buffer_interval = buffer_interval
        self.record_interval = record_interval
        self.record_duration = record_duration

        self.num_test_examples_before = num_test_examples_before
        self.num_training_examples = num_training_examples
        self.num_test_examples_after = num_test_examples_after
        self.total_examples = (
            num_test_examples_before + num_training_examples + num_test_examples_after
        )

        self.output_dir = os.path.join(base_output_dir, run_name)

        if os.path.exists(self.output_dir):
            if overwrite:
                shutil.rmtree(self.output_dir)  # 🔥 delete everything in the folder
            else:
                raise FileExistsError(
                    f"Output directory '{self.output_dir}' already exists. Use overwrite=True to overwrite.")

        os.makedirs(self.output_dir)

        self.metrics_path = os.path.join(self.output_dir, "metrics.npz")
        self.reset_logs()

    def reset_logs(self):
        self.metric_buffers = defaultdict(list)
        self.mean_metrics = defaultdict(lambda: defaultdict(list))
        self.evaluation_steps = []

    def get_phase(self, step):
        if step <= self.num_test_examples_before:
            return "test_before"
        elif step > self.num_test_examples_before + self.num_training_examples:
            return "test_after"
        else:
            return "train"

    def train(self):
        print("[TRAINING]")
        start_time = time.time()

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.train_dataset, start=1):
                if step > self.total_examples:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.reshape(-1, 1)
                targets = targets.reshape(-1, 1)

                phase = self.get_phase(step)
                if phase.startswith("test"):
                    self.model.prediction_update(inputs, dt=self.dt)
                else:
                    self.model.teaching_update(inputs, targets, dt=self.dt)

                prediction = self.model.layers[-1].event_rate.detach().clone()
                loss = self.loss_fn(prediction.reshape(-1, 1), targets.reshape(-1, 1)).item()

                self.metric_buffers["loss"].append(loss)

                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, "burst_prob") and layer.burst_prob.numel() > 1:
                        var = layer.burst_prob.var().item()
                        self.metric_buffers[f"bp_var_layer{i}"].append(var)

                if step % self.buffer_interval == 0:
                    for key, values in self.metric_buffers.items():
                        mean_val = sum(values) / len(values)
                        self.mean_metrics[phase][key].append(mean_val)

                    self.metric_buffers.clear()

                    # Compute ETA
                    elapsed = time.time() - start_time
                    progress = step / self.total_examples
                    total_estimated_time = elapsed / progress if progress > 0 else 0
                    remaining = total_estimated_time - elapsed
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

                    mean_loss = self.mean_metrics[phase]['loss'][-1] if self.mean_metrics[phase]['loss'] else float('nan')
                    print(f"[{phase.upper()}] Step {step}: Mean Loss = {mean_loss:.6f} | ETA: {eta_str}")

                if (step - 1) % self.record_interval == 0:
                    self.run_evaluation(step)

        self.save_metrics()

    def run_evaluation(self, step):
        print("[EVALUATION]")
        self.test_dataset.reset()
        # eval_model = copy.deepcopy(self.model).to(self.device)

        eval_model = MAContinuousBurstCCNNetwork(
            n_inputs=n_inputs,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_outputs=n_outputs,
            p_baseline=0.5,
            tau_W=1000.0,
            lr=0.0
        )
        eval_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

        losses = []
        outputs = []
        targets = []

        layer_data = {
            f"layer{i}_{attr}": []
            for i, layer in enumerate(eval_model.layers)
            for attr in ("event_rate", "burst_rate", "burst_prob")
            if hasattr(layer, attr)
        }

        # test_dataset = self.dataset_fn()
        with torch.no_grad():
            for i, (inp, tgt) in enumerate(self.test_dataset):
                if i >= self.record_duration:
                    break
                inp = inp.to(self.device).reshape(-1, 1)
                tgt = tgt.to(self.device).reshape(-1, 1)
                eval_model.teaching_update(inp, tgt, dt=self.dt)
                out = eval_model.layers[-1].event_rate
                loss = self.loss_fn(out.reshape(-1, 1), tgt.reshape(-1, 1)).item()
                losses.append(loss)
                outputs.append(out.cpu().numpy())
                targets.append(tgt.cpu().numpy())

                # Capture first 10 units of each tracked variable from each layer
                for i, layer in enumerate(eval_model.layers):
                    for attr in ("event_rate", "burst_rate", "burst_prob"):
                        if hasattr(layer, attr):
                            val = getattr(layer, attr)[:10].detach().cpu().numpy()
                            layer_data[f"layer{i}_{attr}"].append(val)

        self.evaluation_steps.append(step)

        checkpoint_path = os.path.join(self.output_dir, f"model_step_{step}.pt")
        torch.save(eval_model.state_dict(), checkpoint_path)

        eval_data_path = os.path.join(self.output_dir, f"eval_step_{step}.npz")
        np.savez(eval_data_path,
                 step=step,
                 mean_test_loss=np.mean(losses),
                 outputs=np.stack(outputs),
                 targets=np.stack(targets),
                 **{k: np.stack(v) for k, v in layer_data.items()}  # add all layer data
        )

        print(f"Saved evaluation at step {step} to:")
        print(f"  {checkpoint_path}")
        print(f"  {eval_data_path}")

    def save_metrics(self):
        mean_metrics_np = {
            phase: {metric: np.array(values) for metric, values in metrics.items()}
            for phase, metrics in self.mean_metrics.items()
        }

        np.savez(self.metrics_path,
                 evaluation_steps=np.array(self.evaluation_steps),
                 mean_metrics=mean_metrics_np)
        print(f"Saved training metrics to '{self.metrics_path}'")


if __name__ == "__main__":
    model_type = 'sin'
    if model_type == 'catcam':
        seed = 1
        set_global_seed(seed)

        # run_name = "catcam_continuous_burstccn_norm_2_lr_10_250_init_1_5_new_ets"
        # target_net_file = Path(__file__).resolve().parent / "saved_models" / "target_network_xavier_2_0_250.pt"
        # normalised_inputs = True
        # downsample_factor = 2
        # n_hidden_units = 250
        # lr = 10.0
        # init_gain = 1.5

        run_name = f"catcam_continuous_burstccn_norm_2_lr_10_250_init_1_75_new_ets_seed{seed}"
        target_net_file = Path(__file__).resolve().parent / "saved_models" /  "target_network_xavier_2_0_250.pt"
        normalised_inputs = True
        downsample_factor = 2
        n_hidden_units = 250
        lr = 10.0
        init_gain = 1.75


        n_inputs = 32 * 32
        n_hidden_layers = 2
        n_outputs = 10

        # lr = 10.0
        tau_W = 1000.0
        dt = 0.01
        alpha = dt / tau_W

        buffer_interval = 1000
        record_interval = 500000
        record_duration = 40000
        num_test_examples_before = 500000
        num_training_examples = 10000000 # 10000000
        num_test_examples_after = 500000

        def make_dataset():
            return CatCamContinuousDataLoader("E:/CatCam", ann_path=target_net_file,
                                              normalised_inputs=True, downsample_factor=downsample_factor)

        train_dataset = make_dataset()

        model = MAContinuousBurstCCNNetwork(
            n_inputs=n_inputs,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_outputs=n_outputs,
            p_baseline=0.5,
            tau_W=tau_W,
            lr=lr
        )

        if normalised_inputs:
            for layer in model.layers:
                torch.nn.init.xavier_normal_(layer.weight, gain=init_gain)
                torch.nn.init.constant_(layer.bias, val=0.0)
            # target_net = ANN(32 ** 2, 2, 500, 10)
            # target_net.load_state_dict(torch.load(target_net_file))
            # model.set_weights(
            #     [layer.weight.clone() for layer in target_net.linear_layers],
            #     [layer.bias.clone() for layer in target_net.linear_layers]
            # )

        else:
            model.set_weights(
                layer_weights=[2 * model.layers[-3].weight.data,
                               3 * model.layers[-2].weight.data,
                               4 * model.layers[-1].weight.data],
                layer_biases=[-0.75 * torch.ones_like(layer.bias.data).squeeze(1) for layer in model.layers]
            )

    elif model_type == 'sin':
        seed = 4

        run_name = f"sin_wave_seed{seed}"
        # target_net_file = r'C:\Users\willg\PycharmProjects\burstccn\continuous_burstccn\saved_models\'

        lr = 2000.0
        init_gain = 1.5

        n_inputs = 3
        n_hidden_layers = 1
        n_hidden_units = 25
        n_outputs = 1

        # lr = 10.0
        tau_W = 1000.0
        dt = 0.01

        buffer_interval = 1000
        record_interval = 10000
        record_duration = 5000
        num_test_examples_before = 5000
        num_training_examples = 200000
        num_test_examples_after = 5000

        def make_dataset():
            set_global_seed(seed)
            return SinWaveContinuousDataLoader(n_inputs=n_inputs, n_hidden_layers=n_hidden_layers,
                                               n_hidden_units=n_hidden_units, n_outputs=n_outputs,
                                               seed=seed)

        train_dataset = make_dataset()
        test_dataset = make_dataset()

        model = MAContinuousBurstCCNNetwork(
            n_inputs=n_inputs,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_outputs=n_outputs,
            p_baseline=0.5,
            tau_W=tau_W,
            lr=lr
        )

        for layer in model.layers:
            torch.nn.init.xavier_normal_(layer.weight, gain=init_gain)
            torch.nn.init.constant_(layer.bias, val=0.0)

    else:
        raise NotImplementedError(f"This model type is not supported: {model_type}")

    trainer = ContinuousModelTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        run_name=run_name,
        dt=dt,
        buffer_interval=buffer_interval,
        record_interval=record_interval,
        record_duration=record_duration,
        num_test_examples_before=num_test_examples_before,
        num_training_examples=num_training_examples,
        num_test_examples_after=num_test_examples_after,
        base_output_dir="training_runs",
        overwrite=True
    )

    trainer.train()
