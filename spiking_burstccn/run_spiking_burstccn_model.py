import pickle
from pathlib import Path

import numpy as np

DEFAULT_SPIKING_DATA_FOLDER = Path(__file__).resolve().parent / "spiking_model_data2"


class SpikingNeuronEnsemble:
    def __init__(self, n_neurons):
        self.neurons = [SpikingNeuron(self, neuron_index=i) for i in range(n_neurons)]

        self.spike_rate_time_constant = 1000e-3  # ms // tau_s
        self.event_rate_time_constant = 1000e-3  # ms // tau_e
        self.burst_rate_time_constant = 1000e-3  # ms // tau_b
        self.burst_prob_time_constant = 1000e-3  # ms // tau_p

        self.spike_rate_instant = 0.0
        self.spike_rate_ma = 0.0

        self.event_rate_instant = 0.0
        self.event_rate_ma = 0.0

        self.burst_rate_instant = 0.0
        self.burst_rate_ma = 0.0

        self.burst_prob_indirect = self.neurons[0].baseline_burst_prob
        # self.burst_prob_direct = self.neurons[0].baseline_burst_prob
        # self.burst_prob_direct_ma = self.neurons[0].baseline_burst_prob

        self.Q_inputs = 0.0
        self.Y_inputs = 0.0
        self.dendritic_potentials = 0.0

        self.soma_W_weights = None

        self.Y_weights = None
        self.Q_weights = None

    def update(self, t, dt, prev_layer=None, next_layer=None):
        if next_layer is not None:
            self.Y_weights = 0.75 * 0.70 * 8e2 * next_layer.soma_W_weights / len(next_layer.soma_W_weights)
            self.Q_weights = 0.75 * 0.70 * -8e2 * next_layer.baseline_burst_probs() * next_layer.soma_W_weights / len(
                next_layer.soma_W_weights)

        for neuron in self.neurons:
            neuron.update(t, dt, prev_layer=prev_layer, next_layer=next_layer)

        self.spike_rate_instant = np.mean([n.spike for n in self.neurons]) / dt
        self.spike_rate_ma = self.spike_rate_ma + (dt / self.spike_rate_time_constant) * (
                    self.spike_rate_instant - self.spike_rate_ma)

        self.event_rate_instant = np.mean([n.event for n in self.neurons]) / dt
        self.event_rate_ma = self.event_rate_ma + (dt / self.event_rate_time_constant) * (
                    self.event_rate_instant - self.event_rate_ma)

        self.burst_rate_instant = np.mean([n.burst for n in self.neurons]) / dt
        self.burst_rate_ma = self.burst_rate_ma + (dt / self.burst_rate_time_constant) * (
                    self.burst_rate_instant - self.burst_rate_ma)

        self.burst_prob_indirect = (self.burst_rate_ma / self.event_rate_ma) if self.event_rate_ma != 0.0 else \
        self.neurons[0].baseline_burst_prob
        # self.burst_prob_direct = np.mean([n.burst_probability for n in self.neurons])
        # self.burst_prob_direct = np.mean([n.burst_probability for n in self.neurons])
        # self.burst_prob_direct_ma = self.burst_prob_direct_ma + (dt/self.burst_prob_time_constant)*(self.burst_prob_direct - self.burst_prob_direct_ma)

        self.Q_inputs = np.mean([n.Q_input for n in self.neurons])
        self.Y_inputs = np.mean([n.Y_input for n in self.neurons])
        self.dendritic_potentials = np.mean([n.dendritic_potential for n in self.neurons])

    def set_soma_input_current(self, soma_input_current):
        for neuron in self.neurons:
            neuron.soma_input_current = soma_input_current  # 1e-9 * np.random.randn()

    def set_somatic_bias_current(self, soma_bias_current):
        for neuron in self.neurons:
            neuron.soma_bias_current = soma_bias_current

    def set_dendritic_input_current(self, dendritic_input_current):
        for neuron in self.neurons:
            neuron.dendritic_input_current = dendritic_input_current  # 1e-9 * np.random.randn()

    def events(self):
        return np.array([n.event for n in self.neurons])

    def prev_events(self):
        return np.array([n.prev_event for n in self.neurons])

    def bursts(self):
        return np.array([n.burst for n in self.neurons])

    def prev_bursts(self):
        return np.array([n.prev_burst for n in self.neurons])

    def baseline_burst_probs(self):
        return np.array([n.baseline_burst_prob for n in self.neurons])


class SpikingNeuron:
    def __init__(self, ensemble, neuron_index):
        self.soma_capacitance = 370e-12  # pF // C_s
        self.soma_time_constant = 16e-3  # ms // tau_s
        self.leak_reversal_potential = -70e-3  # mV // E_L
        self.soma_adaptation_time_constant = 100e-3  # ms // tau_ws
        self.soma_spike_triggered_adaptation = 0  # // ws
        self.soma_spike_triggered_adaptation_strength = 200e-12  # pA // b
        self.dendrosomatic_coupling_strength = 1300e-12  # pA // g_s
        self.soma_reset_voltage = -70e-3  # mV // V_r

        self.somatic_potential = -70e-3  # mV // V_s

        self.soma_spike_threshold_reversal_potential = -50e-3  # mV
        self.soma_spike_threshold_time_constant = 27e-3  # ms
        self.soma_spike_threshold = -50e-3  # mV

        # Full model reversal
        self.dendritic_reversal_potential = -38e-3  # mV // E_d
        # Simplified reversal potential
        # self.dendritic_reversal_potential = -57e-3 # mV // E_d

        self.dendritic_nonlinear_scaling = 6e-3  # mV // D_d
        self.dendritic_nonlinearity = lambda x: 1 / (
                    1 + np.exp(-(x - self.dendritic_reversal_potential) / self.dendritic_nonlinear_scaling))

        self.dendritic_capacitance = 170e-12  # pF // C_d
        self.dendritic_time_constant = 7e-3  # ms // tau_d
        self.dendritic_adaptation_time_constant = 30e-3  # ms // tau_wd
        self.dendritic_subthreshold_adaptation = 0  # // w_d
        self.dendritic_subthreshold_adaptation_strength = 13e-9  # nS // a_w # was 30e-9 by mistake
        self.dendritic_nonlinearity_strength = 1200e-12  # pA // g_d
        self.backpropagating_action_potential_strength = 2600e-12  # pA // c_d

        self.dendritic_potential = -70e-3  # mV // V_d
        # self.dendritic_potential = self.dendritic_reversal_potential

        self.soma_input_current = 0  # A // I_s
        self.dendritic_input_current = 0  # A // I_d

        self.soma_bias_current = 0

        self.box_filter_amplitude = 1
        self.box_filter_duration = 2e-3  # ms
        self.box_filter_delay = 0.5e-3  # ms

        self.spike = 0
        self.convolved_spikes = 0
        self.recent_spike_times = []

        # self.soma_W_weights = np.array([1.0])

        self.Q_input = 0.0
        self.Y_input = 0.0

        # Full model terms
        self.burst_threshold_duration = 16e-3  # ms // b_th
        self.burst = 0
        self.in_burst = False
        self.event = 0

        # Simplified model terms
        # self.burst = 0
        # self.event = 0

        # self.baseline_burst_prob = 0.46
        # self.baseline_burst_prob = 0.4925
        self.baseline_burst_prob = 0.375

        self.prev_spike = 0
        self.prev_burst = 0
        self.prev_event = 0

        self.ensemble = ensemble
        self.neuron_index = neuron_index

    def update(self, t, dt=1e-3, prev_layer=None, next_layer=None):
        self.prev_spike = self.spike
        self.prev_event = self.event
        self.prev_burst = self.burst

        self.spike = 0
        self.burst = 0

        # Full model
        self.event = 0

        self.event_rate_ma = 5.0  # 5Hz
        self.event_rate_ma_tau = 200e-3  # 200ms

        # Simplified model
        # self.event = 0

        # excitatory_soma_noise = 40e-12 * (np.random.rand() < 10 * dt) / 0.01
        # inhibitory_soma_noise = 40e-12 * (np.random.rand() < 10 * dt) / 0.01
        excitatory_soma_noise = 300e-12 * (np.random.rand() < 150 * dt) / (150 * dt)
        inhibitory_soma_noise = -300e-12 * (np.random.rand() < 150 * dt) / (150 * dt)

        # excitatory_soma_noise = 0
        # inhibitory_soma_noise = 0

        if prev_layer is None:
            noiseless_soma_input = self.soma_input_current
            # total_soma_input = self.soma_W_weights * self.soma_input_current + excitatory_soma_noise + inhibitory_soma_noise
        else:
            noiseless_soma_input = prev_layer.prev_events()

        total_soma_input = self.ensemble.soma_W_weights[:, self.neuron_index].dot(
            noiseless_soma_input).item() + excitatory_soma_noise + inhibitory_soma_noise

        total_soma_input += self.soma_bias_current

        # Full model equation
        self.somatic_potential = self.somatic_potential + (dt / self.soma_capacitance) * (
                    -(self.soma_capacitance / self.soma_time_constant) * (
                        self.somatic_potential - self.leak_reversal_potential) + self.dendrosomatic_coupling_strength * self.dendritic_nonlinearity(
                self.dendritic_potential) + total_soma_input - self.soma_spike_triggered_adaptation)

        # Simplified model equation (no direct dendritic effect)
        # self.somatic_potential = self.somatic_potential + (dt / self.soma_capacitance) * (-(self.soma_capacitance / self.soma_time_constant) * (self.somatic_potential - self.leak_reversal_potential) + total_soma_input - self.soma_spike_triggered_adaptation)
        # self.burst_probability = self.dendritic_nonlinearity(self.dendritic_potential)

        self.soma_spike_threshold = self.soma_spike_threshold + (dt / self.soma_spike_threshold_time_constant) * (
                    self.soma_spike_threshold_reversal_potential - self.soma_spike_threshold)

        if self.somatic_potential >= self.soma_spike_threshold:
            self.spike = 1
            self.recent_spike_times.append(t)
            self.somatic_potential = self.soma_reset_voltage
            self.soma_spike_threshold += 2e-3

            # Full model burst generation
            if len(self.recent_spike_times) > 1 and t <= self.recent_spike_times[-2] + self.burst_threshold_duration:
                if self.in_burst is False:
                    self.burst = 1
                    self.in_burst = True
                    self.ensemble.soma_W_weights[:,
                    self.neuron_index] += dt * self.ensemble.soma_W_weights_lr * noiseless_soma_input
                    self.ensemble.soma_W_weights[:, self.neuron_index] = np.minimum(
                        self.ensemble.soma_W_weights[:, self.neuron_index], 1.5)
            else:
                self.event = 1
                self.in_burst = False
                self.ensemble.soma_W_weights[:,
                self.neuron_index] -= dt * self.ensemble.soma_W_weights_lr * self.baseline_burst_prob * noiseless_soma_input
                self.ensemble.soma_W_weights[:, self.neuron_index] = np.maximum(
                    self.ensemble.soma_W_weights[:, self.neuron_index], 0)

        #         if self.event_rate_ma <= 1.0:
        #             self.ensemble.soma_W_weights[:, self.neuron_index] += 0.1
        #         if self.event_rate_ma >= 10.0:
        #             self.ensemble.soma_W_weights[:, self.neuron_index] -= 0.1
        #
        # self.event_rate_ma = self.event_rate_ma * (1 - dt / self.event_rate_ma_tau) + self.event * (dt / self.event_rate_ma_tau)

        # # Simplified model burst generation
        # if len(self.recent_spike_times) <= 1 or t > self.recent_spike_times[-2] + self.burst_threshold_duration:
        #     self.event = 1
        #     if np.random.rand() < self.burst_probability:
        #         self.burst = 1
        #         # LTP
        #         self.ensemble.soma_W_weights[:, self.neuron_index] += dt * self.ensemble.soma_W_weights_lr * (1 - self.baseline_burst_prob) * noiseless_soma_input
        #     else:
        #         # LTD
        #         self.ensemble.soma_W_weights[:, self.neuron_index] -= dt * self.ensemble.soma_W_weights_lr * self.baseline_burst_prob * noiseless_soma_input

        self.soma_spike_triggered_adaptation = self.soma_spike_triggered_adaptation + (
                    dt / self.soma_adaptation_time_constant) * (
                                                           -self.soma_spike_triggered_adaptation + self.soma_spike_triggered_adaptation_strength * self.soma_adaptation_time_constant * self.spike)

        # Do box filter convolution
        self.convolved_spikes = 0
        for spike_time in self.recent_spike_times[:]:
            # if spike_time <= t - (self.box_filter_delay + self.box_filter_duration):
            # Remove old spikes
            if spike_time <= t - self.burst_threshold_duration:
                self.recent_spike_times.remove(spike_time)

            if t - (self.box_filter_delay + self.box_filter_duration) <= spike_time <= t - self.box_filter_delay:
                self.convolved_spikes += 1

        excitatory_dendritic_noise = 400e-12 * (np.random.rand() < 300 * dt) / (300 * dt)
        inhibitory_dendritic_noise = -400e-12 * (np.random.rand() < 300 * dt) / (300 * dt)
        # inhibitory_dendritic_noise -= 125e-12
        # inhibitory_dendritic_noise -= 120e-12
        # excitatory_dendritic_noise += -100e-12
        excitatory_dendritic_noise += -10e-12

        # excitatory_dendritic_noise = 100e-12
        # inhibitory_dendritic_noise = 0

        # excitatory_dendritic_noise = 100e-12 * (np.random.rand() < 20 * dt) / (20 * dt)
        # inhibitory_dendritic_noise = -1500e-12 * (np.random.rand() < 20 * dt) / (20 * dt)
        # inhibitory_dendritic_noise = -320e-12 * (np.random.rand() < 30 * dt) / (30 * dt)

        # excitatory_dendritic_noise = 100e-12 * (np.random.rand() < 10 * dt) / 0.01
        # inhibitory_dendritic_noise = -750e-12 * (np.random.rand() < 10 * dt) / 0.01

        if next_layer is None:
            total_dendritic_input = self.dendritic_input_current + excitatory_dendritic_noise + inhibitory_dendritic_noise
        else:
            self.Q_input = self.ensemble.Q_weights[self.neuron_index, :].dot(next_layer.prev_events()).item()
            self.Y_input = self.ensemble.Y_weights[self.neuron_index, :].dot(next_layer.prev_bursts()).item()
            total_dendritic_input = self.Q_input + self.Y_input + excitatory_dendritic_noise + inhibitory_dendritic_noise

        # total_dendritic_input += 100e-12 * (self.baseline_burst_prob - self.ensemble.burst_prob_indirect)

        # Full model dynamics
        self.dendritic_potential = self.dendritic_potential + (dt / self.dendritic_capacitance) * (
                    -(self.dendritic_capacitance / self.dendritic_time_constant) * (
                        self.dendritic_potential - self.leak_reversal_potential) + self.dendritic_nonlinearity_strength * self.dendritic_nonlinearity(
                self.dendritic_potential) + self.backpropagating_action_potential_strength * self.convolved_spikes + total_dendritic_input - self.dendritic_subthreshold_adaptation)
        # Simplified model dynamics
        # self.backpropagating_action_potential_strength = 0.0
        # self.dendritic_nonlinearity_strength = 0.0
        # self.dendritic_potential = self.dendritic_potential + (dt/self.dendritic_capacitance) * (-(self.dendritic_capacitance/self.dendritic_time_constant) * (self.dendritic_potential - self.leak_reversal_potential) + self.dendritic_nonlinearity_strength*self.dendritic_nonlinearity(self.dendritic_potential) + self.backpropagating_action_potential_strength*self.convolved_spikes + total_dendritic_input - self.dendritic_subthreshold_adaptation)

        self.dendritic_subthreshold_adaptation = self.dendritic_subthreshold_adaptation + (
                    dt / self.dendritic_adaptation_time_constant) * (
                                                             -self.dendritic_subthreshold_adaptation + self.dendritic_subthreshold_adaptation_strength * (
                                                                 self.dendritic_potential - self.leak_reversal_potential))


def save_spiking_model_data(data_folder, settings, state_data):
    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    state_data = {k: np.array(v) for k, v in state_data.items()}

    all_data = {
        "settings": settings,
        **state_data
    }

    with open(data_folder / "spiking_model_data.pkl", "wb") as f:
        pickle.dump(all_data, f)


def run_spiking_model(data_folder, seed):
    np.random.seed(seed)
    # task_type = 'increasing'
    # task_type = 'decreasing'

    # data_folder = 'data_post_thesis'
    # load_previous_data = True

    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    task_types = ['increasing', 'decreasing']

    # # No target, no plasticity
    # pre_learning_start_times = [00.0, 50.0]
    # pre_learning_end_times = [20.0, 60.0]
    #
    # # With target + plasticity
    # during_learning_start_times = [20.0, 60.0]
    # during_learning_end_times = [40.0, 80.0]
    #
    # # No target, no plasticity
    # post_learning_start_times = [40.0, 80.0]
    # post_learning_end_times = [50.0, 90.0]
    #
    # experiment_end_time = 100.0

    # No target, no plasticity
    pre_learning_start_times = [00.0, 50.0]
    pre_learning_end_times = [20.0, 60.0]

    # With target + plasticity
    during_learning_start_times = [20.0, 60.0]
    during_learning_end_times = [40.0, 80.0]

    # No target, no plasticity
    post_learning_start_times = [40.0, 80.0]
    post_learning_end_times = [50.0, 90.0]

    experiment_end_time = 100.0
    dt = 1e-3

    state_data = {key: [] for key in [
        "input_spike_rates", "input_event_rates", "input_burst_rates",
        "input_burst_probs_indirect", "input_dendritic_potentials",
        "input_Q_inputs", "input_Y_inputs", "input_soma_input_weights",
        "output_spike_rates", "output_event_rates", "output_burst_rates",
        "output_burst_probs_indirect", "output_dendritic_potentials",
        "input_all_spikes", "input_bursts", "input_events",
        "output_all_spikes", "output_bursts", "output_events",
        "target_rates"
    ]}

    input_soma_input_weights = []

    input_neurons = SpikingNeuronEnsemble(n_neurons=100)
    output_neurons = SpikingNeuronEnsemble(n_neurons=100)

    input_neurons.soma_W_weights = np.array([[0.4] * len(input_neurons.neurons)])

    output_neurons.soma_W_weights = 0.55 * np.array(
        [[4e-8 / len(input_neurons.neurons)] * len(input_neurons.neurons)] * len(output_neurons.neurons))
    # output_neurons.soma_W_weights = np.array([2e-8 / len(input_neurons.neurons)] * len(input_neurons.neurons))
    # output_neurons.soma_W_weights = np.array([0.01e-8 / len(input_neurons.neurons)] * len(input_neurons.neurons))
    output_neurons.soma_W_weights_lr = 0.0

    t = 0.0

    task_index = 0
    while t <= experiment_end_time:
        if task_index < len(task_types) - 1 and t >= pre_learning_start_times[task_index + 1]:
            print(f'Switching task: {task_types[task_index]} --> {task_types[task_index + 1]}')
            task_index += 1

        if task_types[task_index] == 'increasing':
            target_rate = 10.0
            input_neurons.soma_W_weights_lr = 2.3e11
        elif task_types[task_index] == 'decreasing':
            target_rate = 5.0
            input_neurons.soma_W_weights_lr = 2.3e11
        else:
            raise NotImplementedError

        if t - np.floor(t) < dt:
            print(
                f't={np.floor(t)}, e_1={input_neurons.event_rate_ma:.4f}, e_2={output_neurons.event_rate_ma:.4f}, p_1={input_neurons.burst_prob_indirect:.4f}, p_2={output_neurons.burst_prob_indirect:.4f}, w_0={np.mean(input_neurons.soma_W_weights):.4f}')

        input_neurons.set_soma_input_current(200e-12)

        # input_bias_current = -100e-12 # -1.6e-12 * input_neurons.event_rate_instant
        # input_neurons.set_dendritic_input_current(input_bias_current)

        input_soma_bias_current = 50e-12 - 143e-12 * input_neurons.burst_prob_indirect
        input_neurons.set_somatic_bias_current(input_soma_bias_current)

        output_soma_bias_current = 50e-12 - 143e-12 * output_neurons.burst_prob_indirect
        output_neurons.set_somatic_bias_current(output_soma_bias_current)

        output_dendritic_bias_current = 42e-12 - 1.71e-12 * output_neurons.event_rate_instant  # -1.6e-12 * output_neurons.event_rate_instant

        if during_learning_start_times[task_index] <= t <= during_learning_end_times[task_index]:
            # Set target
            if task_types[task_index] == 'increasing':
                # Going up
                output_neurons.set_dendritic_input_current(
                    80e-12 * (target_rate - output_neurons.event_rate_ma) + output_dendritic_bias_current)
            elif task_types[task_index] == 'decreasing':
                # Going down
                output_neurons.set_dendritic_input_current(
                    100e-12 * (target_rate - output_neurons.event_rate_ma) + output_dendritic_bias_current)
        else:
            # output_neurons.set_dendritic_input_current(0.0)
            output_neurons.set_dendritic_input_current(output_dendritic_bias_current)
            # output_neurons.set_dendritic_input_current(300e-12 * (0.5 - input_neurons.burst_prob_direct))

        input_neurons.update(t, dt, next_layer=output_neurons)
        output_neurons.update(t, dt, prev_layer=input_neurons)

        t += dt

        state_data["input_spike_rates"].append(input_neurons.spike_rate_ma)
        state_data["input_event_rates"].append(input_neurons.event_rate_ma)
        state_data["input_burst_rates"].append(input_neurons.burst_rate_ma)

        # input_burst_probs_direct.append(input_neurons.burst_prob_direct_ma)
        state_data["input_burst_probs_indirect"].append(input_neurons.burst_prob_indirect)

        state_data["input_Q_inputs"].append(input_neurons.Q_inputs)
        state_data["input_Y_inputs"].append(input_neurons.Y_inputs)
        state_data["input_dendritic_potentials"].append(input_neurons.dendritic_potentials)
        state_data["input_soma_input_weights"].append(input_neurons.soma_W_weights.copy().mean())

        state_data["output_spike_rates"].append(output_neurons.spike_rate_ma)
        state_data["output_event_rates"].append(output_neurons.event_rate_ma)
        state_data["output_burst_rates"].append(output_neurons.burst_rate_ma)

        # output_burst_probs_direct.append(output_neurons.burst_prob_direct_ma)
        state_data["output_burst_probs_indirect"].append(output_neurons.burst_prob_indirect)
        state_data["output_dendritic_potentials"].append(output_neurons.dendritic_potentials)

        neurons_to_save = 10
        state_data["input_all_spikes"].append([n.spike for n in input_neurons.neurons[:neurons_to_save]])
        state_data["input_bursts"].append([n.burst for n in input_neurons.neurons[:neurons_to_save]])
        state_data["input_events"].append([n.event for n in input_neurons.neurons[:neurons_to_save]])

        state_data["output_all_spikes"].append([n.spike for n in output_neurons.neurons[:neurons_to_save]])
        state_data["output_bursts"].append([n.burst for n in output_neurons.neurons[:neurons_to_save]])
        state_data["output_events"].append([n.event for n in output_neurons.neurons[:neurons_to_save]])

        state_data["target_rates"].append(target_rate)

        # convolved_spikes.append(input_neurons.input_neurons[0].convolved_spikes)
        # somatic_potentials.append(input_neurons.input_neurons[0].somatic_potential)
        # somatic_thresholds.append(input_neurons.input_neurons[0].soma_spike_threshold)
        # somatic_adaptations.append(input_neurons.input_neurons[0].soma_spike_triggered_adaptation)
        # dendritic_adaptations.append(input_neurons.input_neurons[0].dendritic_subthreshold_adaptation)

    settings = {
        "dt": dt,
        "pre_learning_start_times": pre_learning_start_times,
        "pre_learning_end_times": pre_learning_end_times,

        "during_learning_start_times": during_learning_start_times,
        "during_learning_end_times": during_learning_end_times,

        "post_learning_start_times": post_learning_start_times,
        "post_learning_end_times": post_learning_end_times,

        "experiment_end_time": experiment_end_time,

        "leak_reversal_potential": input_neurons.neurons[0].leak_reversal_potential
    }

    save_spiking_model_data(
        data_folder=data_folder,
        settings=settings,
        state_data=state_data
    )

if __name__ == "__main__":
    run_spiking_model(DEFAULT_SPIKING_DATA_FOLDER, seed=42)
