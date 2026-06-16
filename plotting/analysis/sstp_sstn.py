import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from plotting.analysis.results_store_base import ResultsStore, file_cache_decorator


class SSTpSSTnDataResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path="sstp_sstn_data")
        self.csv_path = Path(__file__).parent / "quentin_chevy_data" / "Fig_7c_data.csv"
        sem_dir = Path(__file__).parent / "quentin_chevy_data"
        self.sem_csv_path = sem_dir / "Fig_7e_sems.csv"
        if not self.sem_csv_path.exists():
            self.sem_csv_path = sem_dir / "Fig_7e_sem.csv"

    @file_cache_decorator()
    def get_all_curves(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV file at {self.csv_path}")

        df = pd.read_csv(self.csv_path, skiprows=2, header=None)
        df.columns = [
            "SSTn_late_X", "SSTn_late_Y",
            "SSTn_early_X", "SSTn_early_Y",
            "SSTp_late_X", "SSTp_late_Y",
            "SSTp_early_X", "SSTp_early_Y",
        ]

        def _xy(x_col, y_col):
            arr = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
            return arr[np.argsort(arr[:, 0])]

        return {
            "early": {
                "sstp": _xy("SSTp_early_X", "SSTp_early_Y"),
                "sstn": _xy("SSTn_early_X", "SSTn_early_Y"),
            },
            "late": {
                "sstp": _xy("SSTp_late_X", "SSTp_late_Y"),
                "sstn": _xy("SSTn_late_X", "SSTn_late_Y"),
            },
        }

    def get_conditions(self):
        return ["early", "late"]

    def get_condition_curves(self, condition):
        curves = self.get_all_curves()
        if condition not in curves:
            raise ValueError(f"Unknown condition: {condition}. Available: {self.get_conditions()}")
        return curves[condition]

    def get_curve(self, condition, population):
        pop_map = {
            "sstp": "sstp",
            "sstn": "sstn",
            "p_plus": "sstp",
            "p_minus": "sstn",
            "p+": "sstp",
            "p-": "sstn",
        }
        if population not in pop_map:
            raise ValueError(f"Unknown population: {population}")
        return self.get_condition_curves(condition)[pop_map[population]]

    def get_plot_arrays(self, condition):
        c = self.get_condition_curves(condition)
        sstp = c["sstp"]
        sstn = c["sstn"]
        return {
            "x_sstp": sstp[:, 0],
            "y_sstp": sstp[:, 1],
            "x_sstn": sstn[:, 0],
            "y_sstn": sstn[:, 1],
        }

    @staticmethod
    def _window_stats_from_curve(x, y, center, halfwidth=0.05, n_samples=400):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        xw = np.linspace(center - halfwidth, center + halfwidth, n_samples)
        yw = np.interp(xw, x, y)

        mean = float(np.trapz(yw, xw) / (2.0 * halfwidth))
        sem = float(np.std(yw, ddof=1) / np.sqrt(max(yw.size, 1)))
        return mean, sem

    @file_cache_decorator()
    def get_reference_bar_sems(self, sem_file_mtime_ns=None):
        """
        Load published SEM values for Fig. 7e.
        Returns Cue/Pre-Reward/Reward SEMs in [SSTp, SSTn] order for early/late,
        plus propagated SEM for Reward - Pre-Reward.
        """
        if not self.sem_csv_path.exists():
            raise FileNotFoundError(f"Could not find SEM CSV file at {self.sem_csv_path}")
        if sem_file_mtime_ns is None:
            sem_file_mtime_ns = self.sem_csv_path.stat().st_mtime_ns
        _ = sem_file_mtime_ns  # cache key buster tied to file content updates

        df = pd.read_csv(self.sem_csv_path, header=None)
        if df.shape[0] < 3:
            raise ValueError(f"Unexpected SEM CSV shape {df.shape}; expected at least 3 rows.")

        n_expected = 12 if df.shape[1] >= 12 else 8
        vals = pd.to_numeric(df.iloc[2, :n_expected], errors="coerce").to_numpy(dtype=float)
        if np.isnan(vals).any():
            raise ValueError("SEM CSV contains non-numeric values in the SEM data row.")

        if n_expected == 12:
            cue = {
                "early": np.array([vals[9], vals[3]], dtype=float),   # [SSTp_early, SSTn_early]
                "late": np.array([vals[6], vals[0]], dtype=float),    # [SSTp_late,  SSTn_late]
            }
            pre_reward = {
                "early": np.array([vals[10], vals[4]], dtype=float),  # [SSTp_early, SSTn_early]
                "late": np.array([vals[7], vals[1]], dtype=float),    # [SSTp_late,  SSTn_late]
            }
            reward = {
                "early": np.array([vals[11], vals[5]], dtype=float),  # [SSTp_early, SSTn_early]
                "late": np.array([vals[8], vals[2]], dtype=float),    # [SSTp_late,  SSTn_late]
            }
        else:
            # Backward-compatible layout: Cue/Reward only.
            cue = {
                "early": np.array([vals[6], vals[2]], dtype=float),
                "late": np.array([vals[4], vals[0]], dtype=float),
            }
            pre_reward = {"early": None, "late": None}
            reward = {
                "early": np.array([vals[7], vals[3]], dtype=float),
                "late": np.array([vals[5], vals[1]], dtype=float),
            }

        reward_delta = {"early": None, "late": None}
        if pre_reward["early"] is not None and pre_reward["late"] is not None:
            reward_delta = {
                "early": np.hypot(pre_reward["early"], reward["early"]),
                "late": np.hypot(pre_reward["late"], reward["late"]),
            }

        sems = {"cue": cue, "pre_reward": pre_reward, "reward": reward, "reward_delta": reward_delta}
        return sems

    def get_bar_plot_data(self, bar_type="cue", cue_time=-1.0, halfwidth=0.05, t0=0.0, t1=1.0):
        curves = self.get_all_curves()
        sem_mtime = self.sem_csv_path.stat().st_mtime_ns if self.sem_csv_path.exists() else None
        reference_sems = self.get_reference_bar_sems(sem_file_mtime_ns=sem_mtime)
        bar_type = str(bar_type).lower()

        if bar_type == "cue":
            epos_m, epos_e = self._window_stats_from_curve(curves["early"]["sstp"][:, 0], curves["early"]["sstp"][:, 1], cue_time, halfwidth)
            eneg_m, eneg_e = self._window_stats_from_curve(curves["early"]["sstn"][:, 0], curves["early"]["sstn"][:, 1], cue_time, halfwidth)
            lpos_m, lpos_e = self._window_stats_from_curve(curves["late"]["sstp"][:, 0], curves["late"]["sstp"][:, 1], cue_time, halfwidth)
            lneg_m, lneg_e = self._window_stats_from_curve(curves["late"]["sstn"][:, 0], curves["late"]["sstn"][:, 1], cue_time, halfwidth)

            return {
                "x_labels": ["SSTp", "SSTn"],
                "early_mean": np.array([epos_m, eneg_m], dtype=float),
                "late_mean": np.array([lpos_m, lneg_m], dtype=float),
                "early_err": np.array([epos_e, eneg_e], dtype=float),
                "late_err": np.array([lpos_e, lneg_e], dtype=float),
                "reference_sems": reference_sems,
            }

        if bar_type in {"delta", "reward_delta"}:
            def _delta(curve):
                m0, e0 = self._window_stats_from_curve(curve[:, 0], curve[:, 1], t0, halfwidth)
                m1, e1 = self._window_stats_from_curve(curve[:, 0], curve[:, 1], t1, halfwidth)
                return m1 - m0, float(np.hypot(e1, e0))

            epos_m, epos_e = _delta(curves["early"]["sstp"])
            eneg_m, eneg_e = _delta(curves["early"]["sstn"])
            lpos_m, lpos_e = _delta(curves["late"]["sstp"])
            lneg_m, lneg_e = _delta(curves["late"]["sstn"])

            # Prefer propagated SEMs from published pre-reward/reward SEMs when available.
            delta_err_early = np.array([epos_e, eneg_e], dtype=float)
            delta_err_late = np.array([lpos_e, lneg_e], dtype=float)
            ref_delta = reference_sems.get("reward_delta", {})
            if ref_delta.get("early") is not None and ref_delta.get("late") is not None:
                delta_err_early = np.asarray(ref_delta["early"], dtype=float)
                delta_err_late = np.asarray(ref_delta["late"], dtype=float)

            return {
                "x_labels": ["SSTp", "SSTn"],
                "early_mean": np.array([epos_m, eneg_m], dtype=float),
                "late_mean": np.array([lpos_m, lneg_m], dtype=float),
                "early_err": delta_err_early,
                "late_err": delta_err_late,
                "reference_sems": reference_sems,
            }

        raise ValueError("bar_type must be one of: 'cue', 'delta', 'reward_delta'")

    def get_data(self, data_id, **kwargs):
        condition = kwargs.get("condition", None)
        population = kwargs.get("population", None)

        if data_id == "all_curves":
            return self.get_all_curves()
        if data_id == "conditions":
            return self.get_conditions()
        if data_id == "condition_curves":
            if condition is None:
                raise ValueError("'condition' is required for data_id='condition_curves'")
            return self.get_condition_curves(condition)
        if data_id == "curve":
            if condition is None or population is None:
                raise ValueError("'condition' and 'population' are required for data_id='curve'")
            return self.get_curve(condition, population)
        if data_id == "plot_arrays":
            if condition is None:
                raise ValueError("'condition' is required for data_id='plot_arrays'")
            return self.get_plot_arrays(condition)
        if data_id == "bar_plot_data":
            return self.get_bar_plot_data(**kwargs)

        raise KeyError("Unknown data_id. Available: ['all_curves', 'conditions', 'condition_curves', 'curve', 'plot_arrays', 'bar_plot_data']")


class SSTpSSTnModelResultsStore(ResultsStore):
    def __init__(self, model_name="mnist_burstccn_dales_4h-0i5zi7qe.h5", saved_activities_root=None):
        super().__init__(cache_path="sstp_sstn_model")
        base_root = Path(saved_activities_root) if saved_activities_root is not None else Path(__file__).resolve().parents[2] / "saved_activities"
        self.default_save_path = base_root / model_name

    @staticmethod
    def _get_posneg_stats_and_arrays(
        starting_activity,
        starting_activity_no_teacher,
        ending_activity,
        ending_activity_no_teacher,
        single_example_id=None,
        nan_safe=False,
        pos_fraction=0.4,
        nonresp_fraction=0.25,
    ):
        if not (0.0 < pos_fraction <= 1.0):
            raise ValueError("pos_fraction must be in (0, 1].")
        if not (0.0 < nonresp_fraction <= 1.0):
            raise ValueError("nonresp_fraction must be in (0, 1].")

        mean_fn = np.nanmean if nan_safe else np.mean
        std_fn = np.nanstd if nan_safe else np.std

        def _safe_reduce(arr, fn):
            if arr.size == 0:
                return np.nan
            return float(fn(arr))

        def _nanmean_no_warn(arr, axis=0):
            arr = np.asarray(arr, dtype=float)
            valid = ~np.isnan(arr)
            count = np.sum(valid, axis=axis)
            total = np.nansum(arr, axis=axis)
            out_shape = total.shape if hasattr(total, "shape") else ()
            out = np.full(out_shape, np.nan, dtype=float)
            return np.divide(total, count, out=out, where=(count > 0))

        means = {("early", "pos"): [], ("early", "nonresp"): [], ("late", "pos"): [], ("late", "nonresp"): []}
        stds = {("early", "pos"): [], ("early", "nonresp"): [], ("late", "pos"): [], ("late", "nonresp"): []}

        early_pos_activities, early_nonresp_activities = [], []
        late_pos_activities, late_nonresp_activities = [], []

        # example_range = range(starting_activity.shape[0]) if single_example_id is None else [single_example_id]
        example_range = range(100)

        for ex in example_range:
            mean_act = 0.0
            e = starting_activity[ex] - mean_act
            ent = starting_activity_no_teacher[ex] - mean_act
            l = ending_activity[ex] - mean_act
            lnt = ending_activity_no_teacher[ex] - mean_act

            activity_score = e - ent
            n_neurons = activity_score.size
            k_pos = max(1, int(np.ceil(pos_fraction * n_neurons)))

            idx_pos = np.argpartition(activity_score, -k_pos)[-k_pos:]
            pos_mask = np.zeros(n_neurons, dtype=bool)
            pos_mask[idx_pos] = True

            nonresp_mask = np.zeros(n_neurons, dtype=bool)
            nonresp_mask[~pos_mask] = True

            overlap = np.count_nonzero(pos_mask & nonresp_mask)
            assert overlap == 0, f"SSTp/SSTn selections overlap ({overlap} neurons)."

            for phase, vals, vals_nt in (("early", e, ent), ("late", l, lnt)):
                for sign, mask in (("pos", pos_mask), ("nonresp", nonresp_mask)):
                    v = vals[mask]
                    vnt = vals_nt[mask]
                    means[(phase, sign)].append([_safe_reduce(vnt, mean_fn), _safe_reduce(v, mean_fn)])
                    stds[(phase, sign)].append([_safe_reduce(vnt, std_fn), _safe_reduce(v, std_fn)])

            early_pos_activities.append(ent[pos_mask])
            early_nonresp_activities.append(ent[nonresp_mask])
            late_pos_activities.append(lnt[pos_mask])
            late_nonresp_activities.append(lnt[nonresp_mask])

        n = len(example_range)
        stats = {}
        for (phase, sign) in means:
            m = np.array(means[(phase, sign)])
            s = np.array(stds[(phase, sign)])
            stats[f"{phase}_{sign}_mean"] = _nanmean_no_warn(m, axis=0)
            # stats[f"{phase}_{sign}_sterr"] = _nanmean_no_warn(s, axis=0) / np.sqrt(n)
            stats[f"{phase}_{sign}_sterr"] = np.std(m, axis=0, ddof=1) / np.sqrt(m.shape[0])

        arrays = {
            "early_pos": np.concatenate(early_pos_activities) if early_pos_activities else np.array([]),
            "early_nonresp": np.concatenate(early_nonresp_activities) if early_nonresp_activities else np.array([]),
            "late_pos": np.concatenate(late_pos_activities) if late_pos_activities else np.array([]),
            "late_nonresp": np.concatenate(late_nonresp_activities) if late_nonresp_activities else np.array([]),
        }
        return stats, arrays

    # @file_cache_decorator()
    def run_analysis(
        self,
        save_path=None,
        layer_idx=0,
        single_example_id=None,
        nan_safe=False,
        pos_fraction=0.4,
        nonresp_fraction=0.25,
    ):
        if save_path is None:
            save_path = self.default_save_path
        save_path = Path(save_path)
        if not save_path.exists():
            raise FileNotFoundError(f"Could not find H5 file at {save_path}")

        with h5py.File(save_path, "r") as f:
            epoch_pattern = re.compile(r"^epoch_(\d+)_layer_\d+_sst_")
            epochs = {int(m.group(1)) for k in f.keys() if (m := epoch_pattern.match(k))}
            if not epochs:
                raise ValueError(f"No matching SST entries found for layer {layer_idx}")
            s_ep, e_ep = min(epochs), max(epochs)

            start = f[f"epoch_{s_ep}_layer_{layer_idx}_sst_with_teacher"][:]
            start_n = f[f"epoch_{s_ep}_layer_{layer_idx}_sst_no_teacher"][:]
            end = f[f"epoch_{e_ep}_layer_{layer_idx}_sst_with_teacher"][:]
            end_n = f[f"epoch_{e_ep}_layer_{layer_idx}_sst_no_teacher"][:]

        stats, arrays = self._get_posneg_stats_and_arrays(
            start,
            start_n,
            end,
            end_n,
            single_example_id=single_example_id,
            nan_safe=nan_safe,
            pos_fraction=pos_fraction,
            nonresp_fraction=nonresp_fraction,
        )

        return {
            "settings": {
                "save_path": str(save_path),
                "layer_idx": int(layer_idx),
                "single_example_id": None if single_example_id is None else int(single_example_id),
                "nan_safe": bool(nan_safe),
                "pos_fraction": float(pos_fraction),
                "nonresp_fraction": float(nonresp_fraction),
            },
            "stats": stats,
            "arrays": arrays,
        }

    def get_conditions(self, **analysis_kwargs):
        _ = analysis_kwargs
        return ["early", "late"]

    def get_condition_curves(self, condition, **analysis_kwargs):
        results = self.run_analysis(**analysis_kwargs)
        stats = results["stats"]
        x = np.array([0.0, 1.0], dtype=float)  # [Stimulus(no_teacher), Error(with_teacher)]

        if condition == "early":
            sstp_mean = np.asarray(stats["early_pos_mean"], dtype=float)
            sstn_mean = np.asarray(stats["early_nonresp_mean"], dtype=float)
            sstp_sterr = np.asarray(stats["early_pos_sterr"], dtype=float)
            sstn_sterr = np.asarray(stats["early_nonresp_sterr"], dtype=float)
        elif condition == "late":
            sstp_mean = np.asarray(stats["late_pos_mean"], dtype=float)
            sstn_mean = np.asarray(stats["late_nonresp_mean"], dtype=float)
            sstp_sterr = np.asarray(stats["late_pos_sterr"], dtype=float)
            sstn_sterr = np.asarray(stats["late_nonresp_sterr"], dtype=float)
        else:
            raise ValueError(f"Unknown condition: {condition}. Available: {self.get_conditions()}")

        return {
            "sstp": np.column_stack([x, sstp_mean]),
            "sstn": np.column_stack([x, sstn_mean]),
            "sstp_sterr": sstp_sterr,
            "sstn_sterr": sstn_sterr,
        }

    def get_curve(self, condition, population, **analysis_kwargs):
        pop_map = {
            "sstp": "sstp",
            "sstn": "sstn",
            "p_plus": "sstp",
            "p_minus": "sstn",
            "p+": "sstp",
            "p-": "sstn",
        }
        if population not in pop_map:
            raise ValueError(f"Unknown population: {population}")
        return self.get_condition_curves(condition, **analysis_kwargs)[pop_map[population]]

    def get_plot_arrays(self, condition, **analysis_kwargs):
        c = self.get_condition_curves(condition, **analysis_kwargs)
        sstp = c["sstp"]
        sstn = c["sstn"]
        return {
            "x_sstp": sstp[:, 0],
            "y_sstp": sstp[:, 1],
            "x_sstn": sstn[:, 0],
            "y_sstn": sstn[:, 1],
            "sstp_sterr": c["sstp_sterr"],
            "sstn_sterr": c["sstn_sterr"],
        }

    def get_bar_plot_data(self, bar_type="cue", **analysis_kwargs):
        stats = self.get_data("stats", **analysis_kwargs)
        bar_type = str(bar_type).lower()

        epos_m = np.asarray(stats["early_pos_mean"], dtype=float)
        eneg_m = np.asarray(stats["early_nonresp_mean"], dtype=float)
        lpos_m = np.asarray(stats["late_pos_mean"], dtype=float)
        lneg_m = np.asarray(stats["late_nonresp_mean"], dtype=float)
        epos_e = np.asarray(stats["early_pos_sterr"], dtype=float)
        eneg_e = np.asarray(stats["early_nonresp_sterr"], dtype=float)
        lpos_e = np.asarray(stats["late_pos_sterr"], dtype=float)
        lneg_e = np.asarray(stats["late_nonresp_sterr"], dtype=float)

        if bar_type == "cue":
            return {
                "x_labels": ["SSTp", "SSTn"],
                "early_mean": np.array([epos_m[0], eneg_m[0]], dtype=float),
                "late_mean": np.array([lpos_m[0], lneg_m[0]], dtype=float),
                "early_err": np.array([epos_e[0], eneg_e[0]], dtype=float),
                "late_err": np.array([lpos_e[0], lneg_e[0]], dtype=float),
            }

        if bar_type in {"delta", "error_delta"}:
            return {
                "x_labels": ["SSTp", "SSTn"],
                "early_mean": np.array([epos_m[1] - epos_m[0], eneg_m[1] - eneg_m[0]], dtype=float),
                "late_mean": np.array([lpos_m[1] - lpos_m[0], lneg_m[1] - lneg_m[0]], dtype=float),
                "early_err": np.array([np.hypot(epos_e[1], epos_e[0]), np.hypot(eneg_e[1], eneg_e[0])], dtype=float),
                "late_err": np.array([np.hypot(lpos_e[1], lpos_e[0]), np.hypot(lneg_e[1], lneg_e[0])], dtype=float),
            }

        raise ValueError("bar_type must be one of: 'cue', 'delta', 'error_delta'")

    def get_data(self, data_id, **analysis_kwargs):
        condition = analysis_kwargs.pop("condition", None)
        population = analysis_kwargs.pop("population", None)

        if data_id == "all_curves":
            return {
                c: self.get_condition_curves(c, **analysis_kwargs)
                for c in self.get_conditions(**analysis_kwargs)
            }
        if data_id == "conditions":
            return self.get_conditions(**analysis_kwargs)
        if data_id == "condition_curves":
            if condition is None:
                raise ValueError("'condition' is required for data_id='condition_curves'")
            return self.get_condition_curves(condition, **analysis_kwargs)
        if data_id == "curve":
            if condition is None or population is None:
                raise ValueError("'condition' and 'population' are required for data_id='curve'")
            return self.get_curve(condition, population, **analysis_kwargs)
        if data_id == "plot_arrays":
            if condition is None:
                raise ValueError("'condition' is required for data_id='plot_arrays'")
            return self.get_plot_arrays(condition, **analysis_kwargs)
        if data_id == "bar_plot_data":
            return self.get_bar_plot_data(**analysis_kwargs)

        results = self.run_analysis(**analysis_kwargs)
        if data_id in results:
            return results[data_id]
        raise KeyError(
            "Unknown data_id. Available: ['all_curves', 'conditions', 'condition_curves', 'curve', 'plot_arrays', 'bar_plot_data', "
            "'settings', 'stats', 'arrays']"
        )


if __name__ == "__main__":
    data_store = SSTpSSTnDataResultsStore()
    model_store = SSTpSSTnModelResultsStore()

    print("Data conditions:", data_store.get_conditions())
    print("Model conditions:", model_store.get_conditions())
    model_store.run_analysis()
