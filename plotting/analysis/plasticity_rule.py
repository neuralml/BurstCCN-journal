import numpy as np

from plotting.analysis.results_store_base import ResultsStore, file_cache_decorator


class PlasticityRuleResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path='plasticity_rule_results_cache.pkl')

    @staticmethod
    def _mean_and_sem(values):
        values = np.asarray(values, dtype=float)
        mean = np.mean(values, axis=0)
        if values.shape[0] > 1:
            sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        else:
            sem = np.zeros_like(mean)
        return mean, sem

    @file_cache_decorator()
    def calculate_plasticity_rule_results(self, rate_start=1.0, rate_stop=80.0, rate_steps=80, duration=100.0, alpha=1.0, seeds=5):
        poisson_rates = np.linspace(rate_start, rate_stop, rate_steps)

        all_weight_changes = []
        all_event_rates = []
        all_burst_rates = []
        all_single_spike_rates = []
        all_burst_probabilites = []

        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            time = 0
            weight_changes = []

            event_counts = []
            burst_counts = []
            single_spike_counts = []
            for rate in poisson_rates:
                spike_number = int(rate * duration)
                isis = rng.exponential(1.0 / rate, spike_number)

                spike_times = []
                event_times = []
                burst_times = []
                single_spike_times = []

                total_weight_change = 0.0
                eta = 0.00001 * alpha
                in_burst = False
                for isi in isis:
                    time += isi

                    if isi > 0.016:
                        in_burst = False
                        event_times.append(time)
                        single_spike_times.append(time)
                        total_weight_change -= eta * 0.5 * rate
                    elif not in_burst:
                        in_burst = True
                        burst_times.append(time)
                        if len(single_spike_times) != 0:
                            single_spike_times.pop()
                        total_weight_change += eta * rate

                    spike_times.append(time)

                weight_changes.append(total_weight_change)

                event_counts.append(len(event_times))
                burst_counts.append(len(burst_times))
                single_spike_counts.append(len(single_spike_times))

            all_event_rates.append(np.array(event_counts) / duration)
            all_burst_rates.append(np.array(burst_counts) / duration)
            all_single_spike_rates.append(np.array(single_spike_counts) / duration)
            all_burst_probabilites.append(np.array(burst_counts) / np.array(event_counts))
            all_weight_changes.append(weight_changes)

        all_event_rates = np.asarray(all_event_rates, dtype=float)
        all_burst_rates = np.asarray(all_burst_rates, dtype=float)
        all_single_spike_rates = np.asarray(all_single_spike_rates, dtype=float)
        all_burst_probabilites = np.asarray(all_burst_probabilites, dtype=float)
        all_weight_changes = np.asarray(all_weight_changes, dtype=float)

        return {
            "poisson_rate": poisson_rates,
            "event_rate": self._mean_and_sem(all_event_rates),
            "burst_rate": self._mean_and_sem(all_burst_rates),
            "single_spike_rate": self._mean_and_sem(all_single_spike_rates),
            "burst_probability": self._mean_and_sem(all_burst_probabilites),
            "delta_W": self._mean_and_sem(all_weight_changes),
        }

    def get_data(self, data_id):
        data = self.calculate_plasticity_rule_results()
        return data[data_id]
