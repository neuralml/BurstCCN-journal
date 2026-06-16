from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from plotting.analysis.results_store_base import WandbResultsStore


class DaleanBottleneckResultsStore(WandbResultsStore):
    def __init__(self):
        super().__init__(cache_path='mnist')

        self.EPOCH_KEY = "epoch"
        self.BATCH_EPOCH_KEY = "batch/epoch"

        self.TOP1_TEST_ERROR_KEY = "epoch/top1_error/test"
        self.BEST_TOP1_TEST_ERROR_KEY = "epoch/top1_error_best/test"

        self.TOP5_TEST_ERROR_KEY = "epoch/top5_error/test"
        self.BEST_TOP5_TEST_ERROR_KEY = "epoch/top5_error_best/test"

        self.BATCH_KEY = "batch"
        self.ANGLE_KEYS = {"qy": "batch/angle_QY/global",
                           "fa": "batch/angle_fa/global_hidden",
                           "bp": "batch/angle_bp/global_hidden",
                           "wy": "batch/angle_WY/global"
                           }

    def get_group_params(self, group):
        GROUP_PARAM_SETS = {
            "equal": dict(
                # bottleneck_sizes=[5, 7, 9, 10, 11, 15, 30, 50, 100, 200],
                # bottleneck_sizes=[5, 7, 10, 15, 30, 50, 100, 200],
                bottleneck_sizes=[2, 4, 6, 8, 10, 15, 20, 50],
            ),
            "reduced": dict(
                reduction_layer=["4thlast", "3rdlast", "2ndlast", "last"]
            )
        }

        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "equal":
            bottleneck_size = kwargs['bottleneck_size']
            run_name = f"fmnist_burstccn_dales_sym_5h_eq{bottleneck_size}"
        elif group == "reduced":
            reduction_layer = kwargs['reduction_layer']
            run_name = f"fmnist_burstccn_dales_sym_5h_{reduction_layer}"
        else:
            raise ValueError(f"Invalid group: {group}")

        return run_name

    def get_wandb_group_name(self, group, **kwargs):
        group_dict = {
            'equal': 'fmnist_burstccn_dales_sym_feb1',
            'reduced': 'fmnist_burstccn_dales_sym_feb1',
        }

        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        wandb_run_name = self.get_wandb_run_name(group, **kwargs)
        wandb_group = self.get_wandb_group_name(group, **kwargs)

        return {"run_name": wandb_run_name,
                "group": wandb_group}


class DaleanBottleneckRankResultsStore:
    def __init__(self, results_dir=None):
        repo_root = Path(__file__).resolve().parents[2]
        self.results_dir = Path(results_dir) if results_dir is not None else repo_root / "burstccn_experiments" / "results"
        self.equal_results_path = self.results_dir / "apic_rank_scan_equal.json"
        self.reduced_layer_results_path = self.results_dir / "apic_rank_scan_reduced_layer.json"

    def load_results(self, scan_mode):
        path = self._get_results_path(scan_mode)
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_scan_mode = payload.get("config", {}).get("scan_mode")
        if saved_scan_mode != scan_mode:
            raise ValueError(f"{path} contains scan_mode={saved_scan_mode!r}, expected {scan_mode!r}.")

        results = payload["results"]
        for result in results:
            result["metrics_by_layer"] = {
                int(layer): metrics for layer, metrics in result["metrics_by_layer"].items()
            }
        return payload["config"], self._sort_results(results)

    def get_equal_apical_pr(self):
        _, results = self.load_results("equal")
        return [
            result for result in self._aggregate_results(results)
            if not result.get("is_baseline")
        ]

    def get_reduced_layer_apical_pr_heatmap(self):
        _, results = self.load_results("reduced_layer")
        results = self._aggregate_results(results)

        baseline_result = next(result for result in results if result.get("is_baseline"))
        plotted_results = [result for result in results if not result.get("is_baseline")]
        measured_layers = sorted({layer for result in results for layer in result["metrics_by_layer"]})
        reduced_layers = sorted(result["scan_value"] for result in plotted_results)
        result_by_reduced_layer = {result["scan_value"]: result for result in plotted_results}

        heatmap = np.full((len(measured_layers), len(reduced_layers)), np.nan, dtype=float)
        for x_idx, reduced_layer in enumerate(reduced_layers):
            result = result_by_reduced_layer[reduced_layer]
            for y_idx, measured_layer in enumerate(measured_layers):
                baseline_value = baseline_result["metrics_by_layer"].get(measured_layer, {}).get("fa_pr")
                current_value = result["metrics_by_layer"].get(measured_layer, {}).get("fa_pr")
                if baseline_value is None or current_value is None or abs(baseline_value) <= 1e-12:
                    continue
                heatmap[y_idx, x_idx] = 100.0 * current_value / baseline_value

        return {
            "heatmap": heatmap,
            "reduced_layers": [layer + 1 for layer in reduced_layers],
            "measured_layers": [layer + 1 for layer in measured_layers],
        }

    def _get_results_path(self, scan_mode):
        if scan_mode == "equal":
            return self.equal_results_path
        if scan_mode == "reduced_layer":
            return self.reduced_layer_results_path
        raise ValueError(f"Unknown scan mode: {scan_mode}")

    def _sort_results(self, results):
        def key(result):
            scan_value, is_baseline = result["case_key"]
            return (1 if is_baseline else 0, scan_value if scan_value is not None else 0, result["seed"])

        return sorted(results, key=key)

    def _aggregate_results(self, results):
        aggregated = []
        by_case_key = defaultdict(list)
        for result in results:
            by_case_key[tuple(result["case_key"])].append(result)

        def case_sort_key(case_key):
            scan_value, is_baseline = case_key
            return (1 if is_baseline else 0, scan_value if scan_value is not None else 0)

        for case_key in sorted(by_case_key, key=case_sort_key):
            group = by_case_key[case_key]
            first = group[0]
            all_layers = sorted({layer for result in group for layer in result["metrics_by_layer"]})
            metrics_by_layer = {}
            sem_by_layer = {}

            for layer in all_layers:
                values = [
                    result["metrics_by_layer"][layer]["fa_pr"]
                    for result in group
                    if "fa_pr" in result["metrics_by_layer"].get(layer, {})
                    and not math.isnan(result["metrics_by_layer"][layer]["fa_pr"])
                ]
                if not values:
                    continue
                metrics_by_layer[layer] = {"fa_pr": float(np.mean(values))}
                sem_by_layer[layer] = {
                    "fa_pr": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                }

            aggregated.append(
                {
                    "case_key": case_key,
                    "scan_value": first["scan_value"],
                    "scan_label": first["scan_label"],
                    "bottlenecks": first["bottlenecks"],
                    "is_baseline": first["is_baseline"],
                    "metrics_by_layer": metrics_by_layer,
                    "sem_by_layer": sem_by_layer,
                    "n_runs": len(group),
                }
            )

        return aggregated


if __name__ == "__main__":
    res = DaleanBottleneckResultsStore()
    print("Equal params:", res.get_group_params("equal"))
    print("Reduced params:", res.get_group_params("reduced"))
