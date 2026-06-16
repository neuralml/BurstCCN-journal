import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from plotting.analysis.results_store_base import ResultsStore, file_cache_decorator


DEFAULT_SPIKING_DATA_FOLDER = Path(__file__).resolve().parents[2] / "spiking_burstccn" / "spiking_model_data2"


class StationaryTargetSpikingResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path='stationary_target_spiking_results_cache.pkl')

    def _generate_plot_metadata(self):
        data_ids = [
            "output_event_rates",
            "output_burst_rates",
            "input_burst_probs_indirect",
            "output_burst_probs_indirect",
            "target_rates",
            "input_Q_inputs",
            "input_Y_inputs",
            "input_dendritic_potentials",
            "output_dendritic_potentials",
            "input_soma_input_weights",
            "input_events",
            "input_bursts"
        ]

        metadata = []
        for data_id in data_ids:
            metadata.append({
                "data_id": data_id,
            })

        return pd.DataFrame(metadata)

    # @file_cache_decorator
    def get_spiking_results(self, data_folder=DEFAULT_SPIKING_DATA_FOLDER):
        data_folder = Path(data_folder)
        with open(data_folder / "spiking_model_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    def get_settings(self):
        data = self.get_spiking_results()
        return data["settings"]

    def get_data(self, data_id, smoothing_sigma=None):
        data = self.get_spiking_results()
        data = data[data_id]
        if smoothing_sigma is not None:
            data = gaussian_filter1d(data, sigma=smoothing_sigma)

        return data
