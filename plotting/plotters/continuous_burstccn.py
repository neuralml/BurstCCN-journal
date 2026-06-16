import numpy as np

from plotting.analysis.continuous_burstccn import ContinuousBurstCCNResultsStore
from plotting.plot_specs.continuous_burstccn import ContinuousBurstCCNAxDetailsStore, ContinuousBurstCCNElemDetailsStore
from plotting.utils import setup_axis, plot_line


class ContinuousBurstCCNPlotter:
    def __init__(self):
        self.results = ContinuousBurstCCNResultsStore()
        self.ax_details = ContinuousBurstCCNAxDetailsStore()
        self.elem_details = ContinuousBurstCCNElemDetailsStore()

    def plot_loss(self, ax, task, block_size, seeds, ax_name='loss'):
        loss_line_metadata = self.elem_details.get('loss')
        smoothed_loss_mean, smoothed_loss_mean_stderr = self.results.get_train_loss(task, seeds=seeds)
        dt = 0.01
        time = np.arange(len(smoothed_loss_mean)) * block_size * dt

        loss_line = plot_line(ax, time, smoothed_loss_mean, yerr=smoothed_loss_mean_stderr, **loss_line_metadata.to_kwargs())
        ax.set_yscale("log")

        self.ax_details._apply_condition_overrides(condition=task)
        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

    def plot_output_comparison(self, ax, task, start_step, end_step, neuron_id, start, end, seed, ax_name='outputs_targets'):
        outputs_before = self.results.get_eval_state(task, "outputs", start_step, neuron_id=neuron_id, start=start, end=end, seed=seed)
        outputs_after = self.results.get_eval_state(task, "outputs", end_step, neuron_id=neuron_id, start=start, end=end, seed=seed)
        targets = self.results.get_eval_state(task, "targets", end_step, neuron_id=neuron_id, start=start, end=end, seed=seed)

        dt = 0.01
        ms_conversion = 1000
        time = np.arange(len(outputs_before)) * dt #* ms_conversion

        target_meta = self.elem_details.get('target_event_rate')
        before_meta = self.elem_details.get('before_event_rate')
        after_meta = self.elem_details.get('after_event_rate')

        plot_line(ax, time, targets, **target_meta.to_kwargs())
        plot_line(ax, time, outputs_before, **before_meta.to_kwargs())
        plot_line(ax, time, outputs_after, **after_meta.to_kwargs())

        self.ax_details._apply_condition_overrides(condition=task)
        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

        if task == 'sin_wave':
            ax.legend(loc='lower right')
        elif task == 'catcam':
            ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.15))


    def plot_burst_probability(self, ax, task, start_step, end_step, neuron_id, start, end, seed, ax_name='burst_probability'):
        bp_before = self.results.get_eval_state(task, "output_burst_prob", start_step, neuron_id=neuron_id, start=start, end=end, seed=seed)
        bp_after = self.results.get_eval_state(task, "output_burst_prob", end_step, neuron_id=neuron_id, start=start, end=end, seed=seed)

        bp_before *= 100
        bp_after *= 100

        dt = 0.01
        ms_conversion = 1000
        time = np.arange(len(bp_before)) * dt #* ms_conversion

        before_meta = self.elem_details.get('before_burst_prob')
        after_meta = self.elem_details.get('after_burst_prob')

        # ax.axhline(y=0.5, color='black', linestyle='dashed')
        ax.axhline(y=50, color='black', linestyle='dashed')
        plot_line(ax, time, bp_before, **before_meta.to_kwargs())
        plot_line(ax, time, bp_after, **after_meta.to_kwargs())

        self.ax_details._apply_condition_overrides(condition=task)
        ax_metadata = self.ax_details.get(ax_name)
        setup_axis(ax, **ax_metadata.to_kwargs())

        if task == 'sin_wave':
            ax.legend(loc='lower right')
        elif task == 'catcam':
            ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.15))
