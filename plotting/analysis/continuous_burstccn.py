import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter, LogFormatterMathtext
from scipy.ndimage import gaussian_filter1d

from plotting.analysis.results_store_base import ResultsStore


class ContinuousBurstCCNResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path=f'continuous_burstccn.pkl')

        self.base_path = Path(__file__).resolve().parents[2] / "continuous_burstccn" / "training_runs"

    def get_run_name(self, task):
        if task == "sin_wave":
            return "sin_wave"
        elif task == "catcam":
            return "catcam_continuous_burstccn_norm_2_lr_10_250_init_1_75_new_ets"
        else:
            raise ValueError(f"Invalid task {task}")

    def get_run_dir(self, task, seed):
        run_name = self.get_run_name(task)

        run_dir = self.base_path / f"{run_name}_seed{seed}"
        return run_dir

    def get_metrics_data(self, task, seed):
        run_dir = self.get_run_dir(task, seed)
        metrics_data = np.load(f"{run_dir}/metrics.npz", allow_pickle=True)
        return metrics_data

    def get_eval_data(self, task, step, seed):
        run_dir = self.get_run_dir(task, seed)
        eval_data = np.load(f"{run_dir}/eval_step_{step}.npz")
        return eval_data

    def get_train_loss(self, task, seeds):
        if task == 'sin_wave':
            sigma = 1.0
        elif task == 'catcam':
            sigma = 40.0
        else:
            raise ValueError(f"Unknown task {task}")

        all_smoothed = []

        for seed in seeds:
            metrics_data = self.get_metrics_data(task, seed)
            mean_metrics = metrics_data["mean_metrics"].item()

            loss_before = mean_metrics["test_before"]["loss"]
            loss_train = mean_metrics["train"]["loss"]

            mean_loss_before = np.mean(loss_before)
            end_mean = np.mean(loss_train[-5:])

            pad_width = int(3 * sigma)

            start_pad = np.full(pad_width, mean_loss_before)
            end_pad = np.full(pad_width, end_mean)

            padded = np.concatenate((start_pad, loss_train, end_pad))
            smoothed_padded = gaussian_filter1d(padded, sigma=sigma, mode="nearest")
            smoothed_loss = smoothed_padded[pad_width:-pad_width]

            all_smoothed.append(smoothed_loss)

        # Stack: shape (n_seeds, n_timesteps)
        all_smoothed = np.stack(all_smoothed)

        mean_loss = np.mean(all_smoothed, axis=0)
        stderr_loss = np.std(all_smoothed, axis=0) / np.sqrt(len(all_smoothed))

        return mean_loss, stderr_loss

    def get_eval_steps(self, task, seed):
        metrics_data = self.get_metrics_data(task, seed)
        steps = metrics_data["evaluation_steps"]
        return steps

    def get_eval_state(self, task, eval_state_key, step, neuron_id, start, end, seed):
        # steps = self.get_eval_steps(run_name)
        eval_data = self.get_eval_data(task, step, seed)

        if eval_state_key == 'output_burst_prob':
            if task == 'sin_wave':
                eval_state_key = 'layer1_burst_prob'
            elif task == 'catcam':
                eval_state_key = 'layer2_burst_prob'

        eval_state = eval_data[eval_state_key]

        return eval_state[start:end, neuron_id]


if __name__ == "__main__":
    # run_name = "catcam_continuous_burstccn_norm_2_lr_3_250_init_2_0_new_ets"
    # run_name = "catcam_continuous_burstccn_norm_2_lr_3_no_output_et"
    # run_name = "catcam_continuous_burstccn_norm_2_lr_1"
    run_name = "catcam_continuous_burstccn_norm_2_lr_6_250_init_1_5_new_ets"


    results = ContinuousBurstCCNResultsStore(run_name=run_name)
    smoothed_loss = results.get_train_loss()

    start_step = 1
    end_step = 10500001
    neuron_id = 7
    start = 200 + 0000
    end = 3200 + 0000

    outputs_before = results.get_eval_state(run_name, "outputs", start_step, neuron_id=neuron_id, start=start, end=end)
    outputs_after = results.get_eval_state(run_name, "outputs", end_step, neuron_id=neuron_id, start=start, end=end)
    targets = results.get_eval_state(run_name, "targets", end_step, neuron_id=neuron_id, start=start, end=end)

    bp_before = results.get_eval_state(run_name, "layer2_burst_prob", start_step, neuron_id=neuron_id, start=start, end=end)
    bp_after = results.get_eval_state(run_name, "layer2_burst_prob", end_step, neuron_id=neuron_id, start=start, end=end)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Plot 1: Smoothed loss (log-scaled y-axis with math formatting and full tick range)
    axes[0].plot(0.1 * np.arange(len(smoothed_loss)), smoothed_loss, color='blue')
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("MSE")

    # Format y-axis with 10^-x notation and ensure 10^-3 is included
    axes[0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    axes[0].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[0].set_ylim(1e-4, max(smoothed_loss) * 1.1)

    # Plot 2: Outputs and targets
    axes[1].plot(np.arange(len(outputs_before)), outputs_before, color='black', label="Before")
    axes[1].plot(np.arange(len(outputs_after)), outputs_after, color='blue', label="After")
    axes[1].plot(np.arange(len(targets)), targets, color='blue', linestyle='dashed', label="Target")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Output")
    axes[1].legend()

    # Plot 3: Burst probability
    axes[2].axhline(y=0.5, color='black', linestyle='dashed')
    axes[2].plot(np.arange(len(bp_before)), bp_before, color='red', label="Before")
    axes[2].plot(np.arange(len(bp_after)), bp_after, color='orange', label="After")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Burst Probability")
    axes[2].legend()

    # Remove top and right spines from all subplots
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
