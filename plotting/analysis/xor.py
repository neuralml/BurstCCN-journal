import pandas as pd
import os
import numpy as np

from plotting.analysis.results_store_base import ResultsStore, file_cache_decorator


class XORResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path='xor_results_cache.pkl')

        self.base_path = r'\\wsl.localhost\Ubuntu-18.04\home\will\spikingburstprop\data'
        self.durex = 8.0
        self.num_realizations = 5

        self.bins_per_example = int(self.durex / 0.4)  # bin_size is 0.4s

        self.er_start_x = -4 * self.bins_per_example
        self.er_end_x = None

        self.bp_start_x = -8 * self.bins_per_example
        self.bp_end_x = -4 * self.bins_per_example

        # self.er_start_x = -32 * self.bins_per_example
        # self.bp_start_x = -64 * self.bins_per_example
        # self.bp_end_x = -32 * self.bins_per_example

        # Load reference time axis
        # self.time_offset = self._get_time_offset()

    # def _generate_plot_metadata(self):
    #     results_map = {
    #             ("burstprop", "one_phase"): "xor_one_phase_5",
    #             ("burstprop", "two_phase"): "xor_two_phase_5",
    #             ("burstccn", "one_phase"): "xor_mod_one_phase_5",
    #             ("burstccn", "two_phase"): "xor_mod_two_phase_5",
    #         }
    #
    #     metadata = []
    #     for (model_type, phase_mode), run_name in results_map.items():
    #         metadata.append({
    #             "model_type": model_type,
    #             "phase_mode": phase_mode,
    #             "run_name": run_name,
    #         })
    #
    #     return pd.DataFrame(metadata)

    def get_run_name(self, model_type, phase_mode):
        results_map = {
                ("burstprop", "one_phase"): "xor_one_phase_5",
                ("burstprop", "two_phase"): "xor_two_phase_5",
                ("burstccn", "one_phase"): "xor_mod_one_phase_5",
                ("burstccn", "two_phase"): "xor_mod_two_phase_5",
            }

        return results_map[(model_type, phase_mode)]

    def _load_concat(self, run_name, data_type, population, start, end):
        prefix = os.path.join(self.base_path, run_name, run_name)

        # Build file prefix and column selector
        if data_type in ("event_rate", "burst_rate"):
            file_template = prefix + f".0.brate_{population}_seed"
            col_index = 2 if data_type == "event_rate" else 1
        elif data_type == "weight_sum":
            file_template = prefix + f".0.wsum_{population}_to_out_seed"
            col_index = "full"
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        data_all = []
        time = None

        for i in range(1, self.num_realizations + 1):
            arr = np.loadtxt(file_template + str(i))[start:end]
            if i == 1:
                time = arr[:, 0]

            if col_index == "full":
                data_all.append(arr[:, 1])  # skip time column
            else:
                data_all.append(arr[:, col_index])

        return time, np.stack(data_all, axis=1)

    @file_cache_decorator()
    def get_data(self, model_type, phase_mode, data_type):
        if data_type == 'output_event_rate':
            return self._get_event_rate_data(model_type, phase_mode, 'output')
        elif data_type == 'output_burst_probability':
            return self._get_burst_probability_data(model_type, phase_mode, 'output')
        elif data_type == 'weight_changes':
            return self._get_weight_change_data(model_type, phase_mode)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _get_event_rate_data(self, model_type, phase_mode, population):
        run_name = self.get_run_name(model_type=model_type, phase_mode=phase_mode)
        # Use event rate-specific window for cleaner slicing
        time, er_data = self._load_concat(run_name,"event_rate", population, self.er_start_x, self.er_end_x)
        # time, er_data = self._load_concat(run_name, "event_rate", population, self.bp_start_x, self.bp_end_x)
        mean_er = np.mean(er_data, axis=1)
        std_er = np.std(er_data, axis=1)
        return time, mean_er, std_er

    # def _get_event_rate_data_from_tr_output(self, population):
    #     all_er_traces = []
    #
    #     for r in range(1, self.num_realizations + 1):
    #         traces = []
    #         for i in range(50):
    #             path = self.prefix + f"{i}.0.tr_output_event_seed{r}"
    #             data = np.loadtxt(path)[self.er_start_x:]
    #             traces.append(data[:, 1])  # column 1 is the event trace value
    #
    #             if i == 0 and r == 1:
    #                 time = data[:, 0]
    #
    #         # shape: (T, 50) → average over neurons
    #         traces = np.stack(traces, axis=1)
    #         all_er_traces.append(np.mean(traces, axis=1))
    #
    #     # shape: (T, num_seeds)
    #     all_er_traces = np.stack(all_er_traces, axis=1)
    #     mean_er = np.mean(all_er_traces, axis=1)
    #     std_er = np.std(all_er_traces, axis=1)
    #
    #     return time, mean_er, std_er

    # def _get_burst_probability_data(self):
    #     brate = self._load_concat(self.prefix + '.0.brate_output_seed', self.bp_start_x, self.bp_end_x,
    #                               drop_first_col=True)
    #     mean_bp = np.mean(brate[:, 0::2] / brate[:, 1::2], axis=1)
    #     std_bp = np.std(brate[:, 0::2] / brate[:, 1::2], axis=1)
    #
    #     # Also get averaged trace from many patterns
    #     all_er = []
    #     all_br = []
    #     for r in range(1, self.num_realizations + 1):
    #         er = np.loadtxt(self.prefix + f'0.0.tr_output_event_seed{r}')[self.bp_start_x:self.bp_end_x]
    #         br = np.loadtxt(self.prefix + f'0.0.tr_output_burst_seed{r}')[self.bp_start_x:self.bp_end_x]
    #         for i in range(1, 50):
    #             er += np.loadtxt(self.prefix + f'{i}.0.tr_output_event_seed{r}')[self.bp_start_x:self.bp_end_x]
    #             br += np.loadtxt(self.prefix + f'{i}.0.tr_output_burst_seed{r}')[self.bp_start_x:self.bp_end_x]
    #         all_er.append(er[:, 1] / 50.)
    #         all_br.append(br[:, 1] / 50.)
    #     all_er = np.array(all_er)
    #     all_br = np.array(all_br)
    #     mean_pbar = np.mean(all_br / all_er, axis=0)
    #     std_pbar = np.std(all_br / all_er, axis=0)
    #
    #     x = self._get_time_axis(self.bp_start_x, self.bp_end_x, offset=-0.4)
    #     return x, mean_bp, std_bp, mean_pbar, std_pbar

    def _get_burst_probability_data(self, model_type, phase_mode, population):
        run_name = self.get_run_name(model_type=model_type, phase_mode=phase_mode)

        # Use event rate-specific window for cleaner slicing
        time, er_data = self._load_concat(run_name, "event_rate", population, self.bp_start_x, self.bp_end_x)
        time, br_data = self._load_concat(run_name,"burst_rate", population, self.bp_start_x, self.bp_end_x)

        bp_data = br_data / er_data

        def exponential_smooth(data, tau, dt):
            alpha = dt / (tau + dt)
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]
            for t in range(1, len(data)):
                smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
            return smoothed

        dt = 0.4  # or whatever your bin size is
        tau = 4.0  # match your plasticity time constant
        if model_type == 'burstccn':
            p_bar = 0.401 * np.ones_like(bp_data)
        elif model_type == 'burstprop':
            p_bar = exponential_smooth(bp_data, tau, dt)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        mean_bp = np.mean(bp_data, axis=1)
        std_bp = np.std(bp_data, axis=1)

        mean_pbar = np.mean(p_bar, axis=1)
        std_pbar = np.std(p_bar, axis=1)

        return time, mean_bp, std_bp, mean_pbar, std_pbar

    def _get_weight_change_data(self, model_type, phase_mode):
        run_name = self.get_run_name(model_type=model_type, phase_mode=phase_mode)

        # burst probability & ERs for computing weight deltas
        x, mean_bp, _, mean_p_bar, _ = self._get_burst_probability_data(model_type, phase_mode, 'output')
        _, mean_er, _ = self._get_event_rate_data(model_type, phase_mode, 'output')

        _, h1_er, _ = self._get_event_rate_data(model_type, phase_mode, 'hidden1')
        _, h2_er, _ = self._get_event_rate_data(model_type, phase_mode, 'hidden2')

        # if model_type == 'burstprop':
        learning_rate_hid_to_out = 4e-3 if phase_mode == 'two_phase' else 4e-4
        bin_size = self.durex / 20

        num_neurons = 2000
        eta = learning_rate_hid_to_out * bin_size / num_neurons

        delta_w1 = num_neurons * eta * (mean_bp - mean_p_bar) * mean_er * h1_er
        delta_w2 = num_neurons * eta * (mean_bp - mean_p_bar) * mean_er * h2_er
        # elif model_type == 'burstccn':
        #     delta_w1 = (mean_bp - 0.401) * mean_er * h1_er
        #     delta_w2 = (mean_bp - 0.401) * mean_er * h2_er
        # else:
        #     raise ValueError(f"Invalid model_type: {model_type}")

        if run_name in ['xor_two_phase_5', 'xor_mod_two_phase_5']:
            mask = np.zeros_like(x, dtype=bool)  # start with all False
            for i in range(4):
                t0 = x[0] + (i + 0.9) * self.durex  # start of plasticity
                t1 = x[0] + (i + 1) * self.durex  # end of example
                mask |= (x >= t0) & (x < t1)
            delta_w1 *= mask
            delta_w2 *= mask

        return x, delta_w1, delta_w2

    # def _get_weight_change_data(self):
    #     # burst probability & ERs for computing weight deltas
    #     x = self._get_time_axis(self.bp_start_x, self.bp_end_x, offset=-0.4)
    #     _, mean_bp, _, _, _ = self._get_burst_probability_data()
    #     _, mean_er, _ = self._get_event_rate_data("output")
    #
    #     hidden1 = self._load_concat(self.prefix + '.0.brate_hidden1_seed', self.bp_start_x, self.bp_end_x,
    #                                 drop_first_col=True)
    #     hidden2 = self._load_concat(self.prefix + '.0.brate_hidden2_seed', self.bp_start_x, self.bp_end_x,
    #                                 drop_first_col=True)
    #     h1_er = np.mean(hidden1[:, 1::2], axis=1)
    #     h2_er = np.mean(hidden2[:, 1::2], axis=1)
    #
    #     delta_w1 = (mean_bp - 0.401) * mean_er * h1_er
    #     delta_w2 = (mean_bp - 0.401) * mean_er * h2_er
    #
    #     return x, delta_w1, delta_w2

    # def _get_weight_change_data_file(self):
    #     start = self.bp_start_x
    #     end = self.bp_end_x
    #
    #     # Load wsum data for hidden1
    #     wsum_hid1 = np.loadtxt(self.prefix + '.0.wsum_hid1_to_out_seed1')[start:end]
    #     wsum_hid2 = np.loadtxt(self.prefix + '.0.wsum_hid2_to_out_seed1')[start:end]
    #
    #     for i in range(2, self.num_realizations + 1):
    #         w1 = np.loadtxt(self.prefix + f'.0.wsum_hid1_to_out_seed{i}')[start:end, 1]
    #         w2 = np.loadtxt(self.prefix + f'.0.wsum_hid2_to_out_seed{i}')[start:end, 1]
    #         wsum_hid1 = np.hstack([wsum_hid1, w1.reshape(-1, 1)])
    #         wsum_hid2 = np.hstack([wsum_hid2, w2.reshape(-1, 1)])
    #
    #     # Compute weight change (difference across timesteps)
    #     dw1 = wsum_hid1[1:, 1:] - wsum_hid1[:-1, 1:]
    #     dw2 = wsum_hid2[1:, 1:] - wsum_hid2[:-1, 1:]
    #
    #     mean_dw1 = np.mean(dw1, axis=1)
    #     mean_dw2 = np.mean(dw2, axis=1)
    #
    #     x = self._get_time_axis(start + 1, end, offset=-0.4)  # x-axis must match diff length
    #
    #     return x, mean_dw1, mean_dw2


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import os


    def moving_average(data, window_size=60):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


    run_names = [
        "xor_one_phase_5",
        "xor_two_phase_5",
        "xor_mod_one_phase_5",
        "xor_mod_two_phase_5"
    ]

    base_path = r"\\wsl.localhost\Ubuntu-18.04\home\will\spikingburstprop\data"

    fig, ax = plt.subplots()

    for run_name in run_names:
        file_path = os.path.join(base_path, run_name, "cost_epoch.dat")
        xor_cost = np.loadtxt(file_path)

        x = xor_cost[:, 0]
        y1 = xor_cost[:, 1]

        window = 3
        if len(x) >= window:
            x_smooth = x[window - 1:]
            y1_smooth = moving_average(y1, window)
            ax.plot(x_smooth, y1_smooth, label=run_name)
        else:
            print(f"Skipping {run_name}: not enough data for window size {window}")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost 1 [smoothed]')
    ax.legend()
    ax.set_ylim([0, 6.5])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


