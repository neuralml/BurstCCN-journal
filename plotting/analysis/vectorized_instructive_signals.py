import json
import csv
from pathlib import Path

import numpy as np

from plotting.analysis.results_store_base import ResultsStore, file_cache_decorator


class VectorizedInstructiveDataResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path='vectorized_instructive_signals_data')
        self.base_path = Path(__file__).parent / "vectorised_instructive_paper_data"
        self.npz_path = self.base_path / "all_curves_decr_incr_error_with_ndnf.npz"
        self.h5_path = self.base_path / "all_curves_decr_incr_error_with_ndnf.h5"
        self.bar_csv_paths = {
            False: {
                "increasing_error": self.base_path / "increasing_error_bar.csv",
                "decreasing_error": self.base_path / "decreasing_error_bar.csv",
            },
            True: {
                "increasing_error": self.base_path / "increasing_error_bar_with_ndnf.csv",
                "decreasing_error": self.base_path / "decreasing_error_bar_with_ndnf.csv",
            },
        }

    @staticmethod
    def _normalise_population(population):
        mapping = {
            "p_minus": "p_minus",
            "p-": "p_minus",
            "minus": "p_minus",
            "p_plus": "p_plus",
            "p+": "p_plus",
            "plus": "p_plus",
        }
        if population not in mapping:
            raise ValueError(f"Invalid population: {population}")
        return mapping[population]

    def _resolve_curve_path(self):
        if self.npz_path.exists():
            return self.npz_path
        if self.h5_path.exists():
            return self.h5_path
        raise FileNotFoundError(f"Neither NPZ nor H5 vectorized instructive data file was found in {self.base_path}")

    @staticmethod
    def _normalise_use_ndnf_data(use_ndnf_data):
        if isinstance(use_ndnf_data, bool):
            return use_ndnf_data
        raise ValueError(f"use_ndnf_data must be a bool, got {use_ndnf_data!r}")

    def _condition_for_ndnf_selection(self, condition, use_ndnf_data):
        use_ndnf_data = self._normalise_use_ndnf_data(use_ndnf_data)
        if use_ndnf_data and not condition.startswith("ndnf_"):
            return f"ndnf_{condition}"
        if not use_ndnf_data and condition.startswith("ndnf_"):
            return condition[len("ndnf_"):]
        return condition

    def _load_from_npz(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Could not find NPZ file at {path}")

        with np.load(path, allow_pickle=True) as npz:
            data = {}

            meta = {}
            if "__meta_json__" in npz.files:
                meta_raw = npz["__meta_json__"]
                if isinstance(meta_raw, np.ndarray):
                    meta_raw = meta_raw.item()
                meta = json.loads(meta_raw)

            if meta.get("conditions"):
                for cond in meta["conditions"]:
                    name = cond["name"]
                    data[name] = {
                        "p_minus": np.asarray(npz[f"{name}__p_minus"], dtype=float),
                        "p_plus": np.asarray(npz[f"{name}__p_plus"], dtype=float),
                        "x_range": tuple(cond.get("x_range", [])),
                        "y_range": tuple(cond.get("y_range", [])),
                    }
            else:
                for key in npz.files:
                    if key == "__meta_json__" or "__" not in key:
                        continue
                    condition, population = key.split("__", 1)
                    data.setdefault(condition, {})[population] = np.asarray(npz[key], dtype=float)

                for condition in data:
                    data[condition].setdefault("x_range", tuple())
                    data[condition].setdefault("y_range", tuple())

        return data

    def _load_from_h5(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Could not find H5 file at {path}")

        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to load vectorized instructive signals from H5") from exc

        data = {}
        with h5py.File(path, "r") as f:
            for condition in f.keys():
                group = f[condition]
                data[condition] = {
                    "p_minus": np.asarray(group["p_minus"][()], dtype=float),
                    "p_plus": np.asarray(group["p_plus"][()], dtype=float),
                    "x_range": tuple(group.attrs.get("x_range", [])),
                    "y_range": tuple(group.attrs.get("y_range", [])),
                }
        return data

    @file_cache_decorator()
    def get_all_curves(self, use_ndnf_data=False):
        _ = self._normalise_use_ndnf_data(use_ndnf_data)
        path = self._resolve_curve_path()
        if path.suffix == ".npz":
            return self._load_from_npz(path)
        if path.suffix == ".h5":
            return self._load_from_h5(path)
        raise ValueError(f"Invalid curve data file extension: {path}")

    def get_conditions(self, use_ndnf_data=False):
        curves = self.get_all_curves(use_ndnf_data=use_ndnf_data)
        prefix = "ndnf_" if self._normalise_use_ndnf_data(use_ndnf_data) else ""
        return sorted(
            condition[len(prefix):] if prefix else condition
            for condition in curves
            if condition.startswith(prefix) and not (not prefix and condition.startswith("ndnf_"))
        )

    def get_condition_curves(self, condition, use_ndnf_data=False, sort_by_x=True):
        curves = self.get_all_curves(use_ndnf_data=use_ndnf_data)
        stored_condition = self._condition_for_ndnf_selection(condition, use_ndnf_data)
        if stored_condition not in curves:
            raise ValueError(f"Unknown condition: {condition}. Available: {self.get_conditions(use_ndnf_data=use_ndnf_data)}")

        condition_curves = dict(curves[stored_condition])

        for population in ["p_minus", "p_plus"]:
            points = np.asarray(condition_curves[population], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(f"Expected shape (N, 2) for {stored_condition}/{population}, got {points.shape}")

            if sort_by_x:
                order = np.argsort(points[:, 0])
                points = points[order]

            condition_curves[population] = points

        return condition_curves

    def get_curve(self, condition, population, use_ndnf_data=False, sort_by_x=True):
        population = self._normalise_population(population)
        condition_curves = self.get_condition_curves(condition=condition, use_ndnf_data=use_ndnf_data, sort_by_x=sort_by_x)
        return condition_curves[population]

    def get_plot_arrays(self, condition, use_ndnf_data=False, sort_by_x=True):
        condition_curves = self.get_condition_curves(condition=condition, use_ndnf_data=use_ndnf_data, sort_by_x=sort_by_x)

        p_minus = condition_curves["p_minus"]
        p_plus = condition_curves["p_plus"]

        return {
            "x_p_minus": p_minus[:, 0],
            "y_p_minus": p_minus[:, 1],
            "x_p_plus": p_plus[:, 0],
            "y_p_plus": p_plus[:, 1],
            "x_range": condition_curves.get("x_range", tuple()),
            "y_range": condition_curves.get("y_range", tuple()),
        }

    @file_cache_decorator()
    def get_bar_plot_data(self, condition, use_ndnf_data=False):
        use_ndnf_data = self._normalise_use_ndnf_data(use_ndnf_data)
        bar_csv_paths = self.bar_csv_paths[use_ndnf_data]
        if condition not in bar_csv_paths:
            raise ValueError(f"Unknown condition: {condition}. Available: {sorted(bar_csv_paths)}")

        csv_path = bar_csv_paths[condition]
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find bar CSV file at {csv_path}")

        with csv_path.open(newline="") as f:
            rows = list(csv.reader(f))

        if len(rows) < 4:
            raise ValueError(f"Expected at least 4 rows in {csv_path.name}, got {len(rows)}")

        mean_row = rows[2]
        mean_plus_err_row = rows[3]

        p_plus_mean = float(mean_row[1])
        p_plus_mean_plus_err = float(mean_plus_err_row[1])
        p_minus_mean = float(mean_row[3])
        p_minus_mean_plus_err = float(mean_plus_err_row[3])

        return {
            "labels": ["P+", "P-"],
            "means": np.array([p_plus_mean, p_minus_mean], dtype=float),
            "errors": np.array([
                abs(p_plus_mean_plus_err - p_plus_mean),
                abs(p_minus_mean_plus_err - p_minus_mean),
            ], dtype=float),
        }

    def get_data(self, data_id, **kwargs):
        use_ndnf_data = kwargs.get("use_ndnf_data", False)
        sort_by_x = kwargs.get("sort_by_x", True)
        condition = kwargs.get("condition", None)
        population = kwargs.get("population", None)

        if data_id == "all_curves":
            return self.get_all_curves(use_ndnf_data=use_ndnf_data)
        if data_id == "conditions":
            return self.get_conditions(use_ndnf_data=use_ndnf_data)
        if data_id == "condition_curves":
            if condition is None:
                raise ValueError("'condition' is required for data_id='condition_curves'")
            return self.get_condition_curves(condition=condition, use_ndnf_data=use_ndnf_data, sort_by_x=sort_by_x)
        if data_id == "curve":
            if condition is None or population is None:
                raise ValueError("'condition' and 'population' are required for data_id='curve'")
            return self.get_curve(condition=condition, population=population, use_ndnf_data=use_ndnf_data, sort_by_x=sort_by_x)
        if data_id == "plot_arrays":
            if condition is None:
                raise ValueError("'condition' is required for data_id='plot_arrays'")
            return self.get_plot_arrays(condition=condition, use_ndnf_data=use_ndnf_data, sort_by_x=sort_by_x)
        if data_id == "bar_plot_data":
            if condition is None:
                raise ValueError("'condition' is required for data_id='bar_plot_data'")
            return self.get_bar_plot_data(condition=condition, use_ndnf_data=use_ndnf_data)

        raise KeyError(
            "Unknown data_id. Available: ['all_curves', 'conditions', 'condition_curves', 'curve', 'plot_arrays', 'bar_plot_data']"
        )



class VectorizedInstructiveModelResultsStore(ResultsStore):
    signal_types = ("burst", "apic", "bp")
    conditions = ("positive_target", "negative_target")
    neuron_types = ("p_plus", "p_minus")
    axes = ("seed", "example", "group_neuron")

    def __init__(self):
        super().__init__(cache_path=None)
        self.base_path = Path(__file__).parent / "vectorised_instructive_paper_data"
        self.data_paths = {
            False: self.base_path / "bci_rotation_prediction_corrected_dalean_no_ndnf.h5",
            True: self.base_path / "bci_rotation_prediction_corrected_dalean_with_ndnf.h5",
        }
        self._file_cache = {}
        self.zscored = {}

    def _resolve_data_path(self, use_ndnf_data=False):
        return self.data_paths[use_ndnf_data]

    def load_all(self, use_ndnf_data=False):
        path = self._resolve_data_path(use_ndnf_data)
        if path in self._file_cache:
            return self._file_cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Could not find Dalean HDF5 file at {path}")

        import h5py

        with h5py.File(path, "r") as f:
            signals = {}
            for signal_type, signal_group in f["signals"].items():
                signals[signal_type] = {}
                for condition, condition_group in signal_group.items():
                    signals[signal_type][condition] = {
                        neuron_type: np.asarray(neuron_group["values"], dtype=float)
                        for neuron_type, neuron_group in condition_group.items()
                    }

            result = {
                "path": path,
                "settings": json.loads(f.attrs.get("settings_json", "{}")),
                "signals": signals,
                "training": {
                    key: np.asarray(dataset)
                    for key, dataset in f.get("training", {}).items()
                },
                "metrics": {
                    key: np.asarray(dataset)
                    for key, dataset in f.get("metrics", {}).items()
                },
            }

        self._file_cache[path] = result
        return result

    def load_state(self, signal_type="burst", condition="positive_target", neuron_type="p_plus", use_ndnf_data=False):
        data = self.load_all(use_ndnf_data=use_ndnf_data)
        return data["signals"][signal_type][condition][neuron_type]

    def zscore_state(
            self,
            signal_type="burst",
            use_ndnf_data=False,
    ):
        cache_key = (use_ndnf_data, signal_type)
        if cache_key in self.zscored:
            return self.zscored[cache_key]

        raw_states = {
            condition: {
                neuron_type: self.load_state(signal_type, condition, neuron_type, use_ndnf_data)
                for neuron_type in self.neuron_types
            }
            for condition in self.conditions
        }
        combined = np.concatenate(
            [
                values.reshape(-1)
                for condition_states in raw_states.values()
                for values in condition_states.values()
            ]
        )
        center = combined.mean()
        scale = combined.std(ddof=0)

        self.zscored[cache_key] = {
            condition: {
                neuron_type: (values - center) / scale
                for neuron_type, values in condition_states.items()
            }
            for condition, condition_states in raw_states.items()
        }
        self.zscored[cache_key]["center"] = center
        self.zscored[cache_key]["scale"] = scale
        return self.zscored[cache_key]

    def zscore_condition(
            self,
            condition="positive_target",
            signal_type="burst",
            use_ndnf_data=False,
    ):
        zscored = self.zscore_state(signal_type, use_ndnf_data)
        return {
            "p_plus": zscored[condition]["p_plus"],
            "p_minus": zscored[condition]["p_minus"],
        }

    @staticmethod
    def _ecdf_on_grid(samples, x_grid):
        samples = np.sort(np.asarray(samples).reshape(-1))
        counts = np.searchsorted(samples, x_grid, side="right")
        return counts / len(samples)

    def calculate_cdf(self, p_plus, p_minus, cdf_grid_points=1000):
        p_plus = np.asarray(p_plus, dtype=float)
        p_minus = np.asarray(p_minus, dtype=float)
        x_grid = np.linspace(
            min(p_plus.min(), p_minus.min()),
            max(p_plus.max(), p_minus.max()),
            int(cdf_grid_points),
        )
        per_neuron_p_plus = np.moveaxis(p_plus, 1, -1).reshape(-1, p_plus.shape[1])
        per_neuron_p_minus = np.moveaxis(p_minus, 1, -1).reshape(-1, p_minus.shape[1])
        p_plus_cdfs = np.stack([self._ecdf_on_grid(neuron_values, x_grid) for neuron_values in per_neuron_p_plus])
        p_minus_cdfs = np.stack([self._ecdf_on_grid(neuron_values, x_grid) for neuron_values in per_neuron_p_minus])
        return {
            "x": x_grid,
            "p_plus": p_plus_cdfs.mean(axis=0),
            "p_plus_stderr": p_plus_cdfs.std(axis=0) / np.sqrt(p_plus_cdfs.shape[0]),
            "p_minus": p_minus_cdfs.mean(axis=0),
            "p_minus_stderr": p_minus_cdfs.std(axis=0) / np.sqrt(p_minus_cdfs.shape[0]),
        }

    def calculate_means(self, condition="positive_target", signal_type="burst", use_ndnf_data=False, zscore=False):
        if zscore:
            states = self.zscore_condition(condition, signal_type, use_ndnf_data)
            p_plus = states["p_plus"]
            p_minus = states["p_minus"]
        else:
            p_plus = self.load_state(signal_type, condition, "p_plus", use_ndnf_data) - 0.5
            p_minus = self.load_state(signal_type, condition, "p_minus", use_ndnf_data) - 0.5

        p_plus_seed_means = p_plus.mean(axis=1).reshape(-1)
        p_minus_seed_means = p_minus.mean(axis=1).reshape(-1)
        return {
            "p_plus": p_plus_seed_means,
            "p_minus": p_minus_seed_means,
        }

    def get_cdf_plot_data(
            self,
            condition,
            use_ndnf_data=False,
            signal_type="burst",
            cdf_grid_points=1000,
            zscore=False,
    ):
        if zscore:
            states = self.zscore_condition(condition, signal_type, use_ndnf_data)
        else:
            states = {
                "p_plus": self.load_state(signal_type, condition, "p_plus", use_ndnf_data),
                "p_minus": self.load_state(signal_type, condition, "p_minus", use_ndnf_data),
            }
        cdf = self.calculate_cdf(states["p_plus"], states["p_minus"], cdf_grid_points)
        return {
            "x_p_plus": cdf["x"],
            "y_p_plus": cdf["p_plus"],
            "x_p_minus": cdf["x"],
            "y_p_minus": cdf["p_minus"],
            "p_plus_stderr": cdf["p_plus_stderr"],
            "p_minus_stderr": cdf["p_minus_stderr"],
            "x_range": (float(cdf["x"].min()), float(cdf["x"].max())),
            "y_range": (0.0, 1.0),
        }


if __name__ == "__main__":
    data_res = VectorizedInstructiveDataResultsStore()
    print("Available data conditions:", data_res.get_conditions())

    model_res = VectorizedInstructiveModelResultsStore()
    model_res
