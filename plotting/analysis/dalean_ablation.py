import json
from pathlib import Path

import numpy as np


class DaleanAblationResultsStore:
    MODEL_RESULTS_PATH = (
        Path(__file__).resolve().parents[2]
        / "burstccn_experiments"
        / "results"
        / "simple_increasing_relative_target_ablation_scan.json"
    )
    MODEL_INITIAL_WEIGHT_KEY = "w_direct_initial"
    MODEL_FINAL_WEIGHT_KEY = "w_direct_final"
    MODEL_EVENT_RATE_CHANGE_KEY = "event_rate_diff"
    MODEL_BURST_PROBABILITY_INITIAL_KEY = "p_t_initial"
    MODEL_LABEL_BY_INTERNEURON_TYPE = {
        "control": "none",
        "pv": "PV-like",
        "vip": "VIP-like",
        "sst": "SST-like",
        "ndnf": "NDNF-like",
    }

    def __init__(self):
        ablation_data_raw = {
            "none": [
                [0.69, 0.80, 1.13, 1.13, 1.45, 1.67, 1.73, 2.76],
                [1.59, 1.88, 1.30, 4.66, 0.90, 2.63, 2.88, 7.06],
            ],
            "PV": [
                [1.82, 2.73, 2.99, 3.36, 3.59, 4.15, 4.32],
                [1.54, 5.04, 2.97, 2.70, 4.62, 3.87, 8.34],
            ],
            "VIP": [
                [0.49, 0.72, 0.91, 0.99, 0.99, 1.47, 2.35],
                [0.49, 0.91, 0.81, 0.63, 0.81, 1.05, 1.74],
            ],
            "SST": [
                [0.72, 0.78, 1.10, 1.39, 1.70, 1.70, 2.82],
                [1.60, 1.78, 4.64, 0.93, 2.63, 2.76, 7.03],
            ],
        }

        self.ablation_data = self._compute_pairwise_changes_dict(ablation_data_raw)
        self._model_payload = self._load_model_payload()
        self.ablation_model = self._load_model_ablation_raw(metric="relative_weight")

    @staticmethod
    def _compute_pairwise_changes(values_a, values_b):
        changes = []
        for pre, post in zip(values_a, values_b):
            # mean_val = (a + b) / 2.0
            # changes.append((b - a) / mean_val)
            # if mean_val == 0:
            #     changes.append(np.nan)
            # else:

            changes.append(100 * (post - pre) / pre)

        return changes

    def _compute_pairwise_changes_dict(self, raw_dict):
        out = {}
        for key, pair in raw_dict.items():
            out[key] = self._compute_pairwise_changes(pair[0], pair[1])
        return out

    @staticmethod
    def _compute_percent_changes(values, pre_value=1.0):
        if pre_value == 0:
            return [np.nan for _ in values]
        return [100.0 * value / pre_value for value in values]

    def _compute_percent_changes_dict(self, raw_dict, pre_value=1.0):
        out = {}
        for key, values in raw_dict.items():
            out[key] = self._compute_percent_changes(values, pre_value=pre_value)
        return out

    @classmethod
    def _load_model_payload(cls):
        if not cls.MODEL_RESULTS_PATH.exists():
            raise FileNotFoundError(f"Could not find ablation scan results: {cls.MODEL_RESULTS_PATH}")
        return json.loads(cls.MODEL_RESULTS_PATH.read_text(encoding="utf-8"))

    def _load_model_ablation_raw(self, metric):
        payload = self._model_payload
        if "interneuron_results" in payload:
            return self._load_model_ablation_from_grouped_results(payload["interneuron_results"], metric=metric)
        if "results" in payload:
            return self._load_model_ablation_from_flat_results(payload["results"], metric=metric)
        raise KeyError(
            f"Ablation scan results must contain 'interneuron_results' or 'results': {self.MODEL_RESULTS_PATH}"
        )

    @classmethod
    def _load_model_ablation_from_grouped_results(cls, interneuron_results, metric):
        out = {}
        for interneuron_type, label in cls.MODEL_LABEL_BY_INTERNEURON_TYPE.items():
            values = interneuron_results.get(interneuron_type, {})
            if metric == "relative_weight":
                out[label] = cls._compute_relative_weight_changes(
                    values.get(cls.MODEL_INITIAL_WEIGHT_KEY, []),
                    values.get(cls.MODEL_FINAL_WEIGHT_KEY, []),
                )
            elif metric == "event_rate_change":
                out[label] = cls._clean_numeric_values(values.get(cls.MODEL_EVENT_RATE_CHANGE_KEY, []))
            elif metric == "burst_probability_change":
                out[label] = cls._compute_initial_burst_probability_offsets(
                    values.get(cls.MODEL_BURST_PROBABILITY_INITIAL_KEY, [])
                )
            else:
                raise ValueError(f"Unknown ablation model metric: {metric}")
        return out

    @classmethod
    def _load_model_ablation_from_flat_results(cls, results, metric):
        out = {label: [] for label in cls.MODEL_LABEL_BY_INTERNEURON_TYPE.values()}
        for result in results:
            label = cls.MODEL_LABEL_BY_INTERNEURON_TYPE.get(result.get("interneuron_type"))
            if label is None:
                continue
            if metric == "relative_weight":
                value = cls._compute_relative_weight_change(
                    result.get(cls.MODEL_INITIAL_WEIGHT_KEY),
                    result.get(cls.MODEL_FINAL_WEIGHT_KEY),
                )
            elif metric == "event_rate_change":
                value = cls._clean_numeric_value(result.get(cls.MODEL_EVENT_RATE_CHANGE_KEY))
            elif metric == "burst_probability_change":
                value = cls._compute_initial_burst_probability_offset(
                    result.get(cls.MODEL_BURST_PROBABILITY_INITIAL_KEY)
                )
            else:
                raise ValueError(f"Unknown ablation model metric: {metric}")
            if value is not None:
                out[label].append(value)
        return out

    @staticmethod
    def _clean_numeric_values(values):
        return [
            clean_value
            for clean_value in (DaleanAblationResultsStore._clean_numeric_value(value) for value in values)
            if clean_value is not None
        ]

    @staticmethod
    def _clean_numeric_value(value):
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    @classmethod
    def _compute_relative_weight_changes(cls, initial_values, final_values):
        return [
            relative_change
            for relative_change in (
                cls._compute_relative_weight_change(initial, final)
                for initial, final in zip(initial_values, final_values)
            )
            if relative_change is not None
        ]

    @staticmethod
    def _compute_relative_weight_change(initial_value, final_value):
        if initial_value is None or final_value is None:
            return None
        initial_value = float(initial_value)
        final_value = float(final_value)
        if not np.isfinite(initial_value) or not np.isfinite(final_value) or initial_value == 0.0:
            return None
        return 100.0 * (final_value - initial_value) / initial_value

    @classmethod
    def _compute_initial_burst_probability_offsets(cls, initial_values):
        return [
            offset
            for offset in (
                cls._compute_initial_burst_probability_offset(initial_value)
                for initial_value in initial_values
            )
            if offset is not None
        ]

    @staticmethod
    def _compute_initial_burst_probability_offset(initial_value):
        initial_value = DaleanAblationResultsStore._clean_numeric_value(initial_value)
        if initial_value is None:
            return None
        return initial_value - 0.5

    def get_ablation_data(self):
        return self.ablation_data

    def get_ablation_model(self):
        return self.ablation_model

    def get_ablation_model_metric(self, metric="relative_weight"):
        if metric == "relative_weight":
            return self.ablation_model
        return self._load_model_ablation_raw(metric=metric)

    def get_ablation_model_event_rate_change(self):
        return self.get_ablation_model_metric(metric="event_rate_change")

    def get_ablation_model_burst_probability_change(self):
        return self.get_ablation_model_metric(metric="burst_probability_change")
