from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import run_command, run_jobs, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

GROUP = "tmp"
RUN_NAME = "fmnist_burstccn_dales_rank_scan"

# MODE = "run"
MODE = "plot"

# SCAN_MODE = "equal"
SCAN_MODE = "reduced_layer"

N_SEEDS = 5
N_PARALLEL = 2

RESULTS_PATH = SCRIPT_DIR / "results" / f"apic_rank_scan_{SCAN_MODE}.json"
EQUAL_RESULTS_PATH = SCRIPT_DIR / "results" / "apic_rank_scan_equal.json"
REDUCED_LAYER_RESULTS_PATH = SCRIPT_DIR / "results" / "apic_rank_scan_reduced_layer.json"
PDF_PATH = SCRIPT_DIR / "plots" / "apic_rank_scan_apical_pr.pdf"

N_SST_VALUES = [2, 4, 6, 8, 10, 15, 20, 50]
N_LAYERS = 4
BASE_SST = 8
REDUCED_SST = 4

TRACKED_METRICS = [
    ("fa_pr", "Apical PR"),
    ("fa_erank", "Apical eRank"),
    ("bp_pr", "BP PR"),
    ("bp_erank", "BP eRank"),
    ("bp_k", "BP k"),
    ("task_energy_ratio", "Task Energy Ratio"),
    ("task_pr", "Task PR"),
    ("task_erank", "Task eRank"),
    ("fa_bp_angle", "Negative-Apical/BP Angle (deg)"),
    ("subspace_overlap", "Subspace Overlap"),
    ("mean_angle", "Mean Angle (deg)"),
    ("max_angle", "Max Angle (deg)"),
]
SUMMARY_METRICS = ["fa_erank", "bp_erank", "task_energy_ratio", "task_erank", "fa_bp_angle", "subspace_overlap"]
TOP1_TEST_ERROR_METRICS = [
    ("epoch/top1_error/test", "final"),
    ("epoch/top1_error_best/test", "best"),
]


def make_bottlenecks(n_sst):
    return [n_sst, n_sst, n_sst, min(10, n_sst)]


def make_reduced_layer_bottlenecks(reduced_layer_index):
    bottlenecks = [BASE_SST] * N_LAYERS
    bottlenecks[reduced_layer_index] = REDUCED_SST
    return bottlenecks


def build_scan_cases():
    if SCAN_MODE == "equal":
        return [
            {
                "scan_value": n_sst,
                "scan_label": str(n_sst),
                "bottlenecks": make_bottlenecks(n_sst),
                "is_baseline": False,
            }
            for n_sst in N_SST_VALUES
        ]

    cases = [
        {
            "scan_value": layer_index,
            "scan_label": str(layer_index + 1),
            "bottlenecks": make_reduced_layer_bottlenecks(layer_index),
            "is_baseline": False,
        }
        for layer_index in range(N_LAYERS)
    ]
    cases.append(
        {
            "scan_value": None,
            "scan_label": "baseline",
            "bottlenecks": [BASE_SST] * N_LAYERS,
            "is_baseline": True,
        }
    )
    return cases


def format_hydra_list(values):
    return "[" + ",".join(str(value) for value in values) + "]"


def cosine_to_negative_apic_angle_deg(cosine_value):
    return math.degrees(math.acos(-max(-1.0, min(1.0, cosine_value))))


def parse_rank_output(output_text):
    metrics_by_layer = {}
    for line in output_text.splitlines():
        line = line.strip()
        if not line.startswith("[RankTrackerResult] "):
            continue
        payload = json.loads(line[len("[RankTrackerResult] ") :])
        layer = int(payload.pop("layer"))
        payload["mean_angle"] = payload.pop("mean_principal_angle_deg")
        payload["max_angle"] = payload.pop("max_principal_angle_deg")
        payload["fa_bp_angle"] = cosine_to_negative_apic_angle_deg(payload["fa_bp_cosine"])
        metrics_by_layer[layer] = payload
    if not metrics_by_layer:
        raise ValueError("No [RankTrackerResult] lines found in training output.")
    return metrics_by_layer


def parse_training_result(output_text):
    for line in output_text.splitlines():
        line = line.strip()
        if line.startswith("[TrainingResult] "):
            return json.loads(line[len("[TrainingResult] ") :])
    raise ValueError("No [TrainingResult] line found in training output.")


def run_one(job):
    case, seed = job
    cmd = [
        sys.executable,
        "train.py",
        "--config-name=mnist_burstccn_dales",
        "dataset=fmnist",
        f"+run_name={RUN_NAME}",
        f"+group={GROUP}",
        "training.n_epochs=3",
        "model.n_hidden_units=50",
        f"training.seed={seed}",
        "+training/triggers@training.triggers.rank_tracker=rank_tracker",
        f"model.feedback_bottleneck_sizes={format_hydra_list(case['bottlenecks'])}",
        "model.Y.mode=sym_W",
        "model.Y.scale=1.0"
    ]

    _, output = run_command(cmd, cwd=REPO_ROOT)
    training_result = parse_training_result(output)
    return {
        "case_key": [case["scan_value"], case["is_baseline"]],
        "scan_value": case["scan_value"],
        "scan_label": case["scan_label"],
        "is_baseline": case["is_baseline"],
        "seed": seed,
        "bottlenecks": case["bottlenecks"],
        **training_result,
        "metrics_by_layer": parse_rank_output(output),
    }


def sort_results(results):
    def key(result):
        scan_value, is_baseline = result["case_key"]
        return (1 if is_baseline else 0, scan_value if scan_value is not None else 0, result["seed"])

    return sorted(results, key=key)


def save_results(results):
    payload = {
        "config": {
            "scan_mode": SCAN_MODE,
            "n_seeds": N_SEEDS,
            "n_parallel": N_PARALLEL,
            "base_sst": BASE_SST,
            "reduced_sst": REDUCED_SST,
            "group": GROUP,
            "run_name": RUN_NAME,
        },
        "results": sort_results(results),
    }
    write_json_atomic(RESULTS_PATH, payload)


def load_results(path, expected_scan_mode=None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_scan_mode = payload.get("config", {}).get("scan_mode")
    if expected_scan_mode is not None and saved_scan_mode != expected_scan_mode:
        raise ValueError(
            f"{path} contains scan_mode={saved_scan_mode!r}, "
            f"but expected {expected_scan_mode!r}."
        )
    results = payload["results"]
    for result in results:
        result["metrics_by_layer"] = {
            int(layer): metrics for layer, metrics in result["metrics_by_layer"].items()
        }
    return payload["config"], sort_results(results)


def aggregate_results(results):
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
        metrics_std_by_layer = {}
        n_successes_by_layer = {}
        training_metrics = {}
        training_metrics_std = {}

        for layer in all_layers:
            metric_names = sorted(
                metric_name
                for result in group
                for metric_name in result["metrics_by_layer"].get(layer, {})
            )
            layer_means = {}
            layer_stds = {}
            layer_counts = {}
            for metric_name in metric_names:
                values = [
                    result["metrics_by_layer"][layer][metric_name]
                    for result in group
                    if metric_name in result["metrics_by_layer"].get(layer, {})
                    and not math.isnan(result["metrics_by_layer"][layer][metric_name])
                ]
                if not values:
                    continue
                layer_means[metric_name] = float(np.mean(values))
                layer_stds[metric_name] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                layer_counts[metric_name] = len(values)
            if layer_means:
                metrics_by_layer[layer] = layer_means
                metrics_std_by_layer[layer] = layer_stds
                n_successes_by_layer[layer] = layer_counts

        for metric_name, _ in TOP1_TEST_ERROR_METRICS:
            values = [
                result[metric_name]
                for result in group
                if result.get(metric_name) is not None and not math.isnan(result[metric_name])
            ]
            if values:
                training_metrics[metric_name] = float(np.mean(values))
                training_metrics_std[metric_name] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        aggregated.append(
            {
                "case_key": case_key,
                "scan_value": first["scan_value"],
                "scan_label": first["scan_label"],
                "bottlenecks": first["bottlenecks"],
                "is_baseline": first["is_baseline"],
                "metrics_by_layer": metrics_by_layer,
                "metrics_std_by_layer": metrics_std_by_layer,
                "training_metrics": training_metrics,
                "training_metrics_std": training_metrics_std,
                "n_successes_by_layer": n_successes_by_layer,
                "n_runs": len(group),
            }
        )
    return aggregated


def render_metric_tables(results, metric_names):
    all_layers = sorted({layer for result in results for layer in result["metrics_by_layer"]})
    tables = []
    for metric_name in metric_names:
        headers = ["scan", "bottlenecks"] + [str(layer + 1) for layer in all_layers]
        rows = []
        for result in results:
            row = [str(result["scan_label"]), str(result["bottlenecks"])]
            for layer in all_layers:
                value = result["metrics_by_layer"].get(layer, {}).get(metric_name)
                row.append("" if value is None or math.isnan(value) else f"{value:.3f}")
            rows.append(row)

        widths = [len(header) for header in headers]
        for row in rows:
            widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

        def format_row(row):
            return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

        title = next((label for name, label in TRACKED_METRICS if name == metric_name), metric_name)
        tables.append(
            "\n".join(
                [f"{title} ({metric_name})", format_row(headers), "-+-".join("-" * width for width in widths)]
                + [format_row(row) for row in rows]
            )
        )
    return "\n\n".join(tables)


def metric_sem(result, layer, metric_name):
    std = result["metrics_std_by_layer"].get(layer, {}).get(metric_name, 0.0)
    count = result["n_successes_by_layer"].get(layer, {}).get(metric_name, 1)
    return std / math.sqrt(count) if count > 1 else 0.0


def plot_equal_apical_pr(ax, results):
    all_layers = sorted({layer for result in results for layer in result["metrics_by_layer"]})
    x_values = [result["scan_value"] for result in results]

    for layer in all_layers:
        xs = []
        ys = []
        yerr = []
        for result in results:
            value = result["metrics_by_layer"].get(layer, {}).get("fa_pr")
            if value is None or math.isnan(value):
                continue
            xs.append(result["scan_value"])
            ys.append(value)
            yerr.append(metric_sem(result, layer, "fa_pr"))
        if ys:
            ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=f"Layer {layer + 1}")

    # ax.set_title("Equal Bottleneck: Apical PR")
    ax.set_xlabel("#SST")
    ax.set_ylabel("Feedback rank")
    ax.set_xticks(x_values)
    # ax.grid(True, alpha=0.3)
    ax.legend()


def plot_reduced_layer_apical_pr_heatmap(fig, ax, results):
    from matplotlib.colors import LinearSegmentedColormap

    baseline_result = next(result for result in results if result.get("is_baseline"))
    plotted_results = [result for result in results if not result.get("is_baseline")]
    all_layers = sorted({layer for result in results for layer in result["metrics_by_layer"]})
    reduced_layers = sorted(result["scan_value"] for result in plotted_results)
    result_by_reduced_layer = {result["scan_value"]: result for result in plotted_results}

    heatmap = np.full((len(all_layers), len(reduced_layers)), np.nan, dtype=float)
    for x_idx, reduced_layer in enumerate(reduced_layers):
        result = result_by_reduced_layer[reduced_layer]
        for y_idx, measured_layer in enumerate(all_layers):
            baseline_value = baseline_result["metrics_by_layer"].get(measured_layer, {}).get("fa_pr")
            current_value = result["metrics_by_layer"].get(measured_layer, {}).get("fa_pr")
            if baseline_value is None or current_value is None or abs(baseline_value) <= 1e-12:
                continue
            heatmap[y_idx, x_idx] = 100.0 * current_value / baseline_value

    finite_values = heatmap[np.isfinite(heatmap)]
    vmin = float(finite_values.min()) if finite_values.size else 0.0
    cmap = LinearSegmentedColormap.from_list("baseline_drop", ["#b30000", "#ffffff"])
    image = ax.imshow(heatmap, aspect="auto", origin="lower", interpolation="nearest", vmin=vmin, vmax=100.0, cmap=cmap)
    # ax.set_title("Reduced Area: Apical PR")
    ax.set_xlabel("Reduced SST area")
    ax.set_ylabel("Measured SST area")
    ax.set_xticks(range(len(reduced_layers)))
    ax.set_xticklabels([layer + 1 for layer in reduced_layers])
    ax.set_yticks(range(len(all_layers)))
    ax.set_yticklabels([layer + 1 for layer in all_layers])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Fraction of original rank [%]")


def plot_apical_pr_summary():
    _, equal_results = load_results(EQUAL_RESULTS_PATH, "equal")
    _, reduced_results = load_results(REDUCED_LAYER_RESULTS_PATH, "reduced_layer")
    equal_results = aggregate_results(equal_results)
    reduced_results = aggregate_results(reduced_results)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), squeeze=False)
    plot_equal_apical_pr(axes[0, 0], equal_results)
    plot_reduced_layer_apical_pr_heatmap(fig, axes[0, 1], reduced_results)
    fig.tight_layout()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    print(f"Saved plot to: {PDF_PATH}")
    plt.show()


def print_result_summary(result):
    summary_parts = []
    for layer, metrics in sorted(result["metrics_by_layer"].items()):
        pieces = []
        for metric_name in ("fa_erank", "task_erank", "task_energy_ratio", "fa_bp_angle"):
            value = metrics.get(metric_name)
            if value is not None and not math.isnan(value):
                suffix = "deg" if metric_name.endswith("angle") else ""
                pieces.append(f"{metric_name}={value:.3f}{suffix}")
        if pieces:
            summary_parts.append(f"L{layer + 1}(" + ", ".join(pieces) + ")")
    test_error = result.get("epoch/top1_error/test")
    best_test_error = result.get("epoch/top1_error_best/test")
    prefix_parts = []
    if test_error is not None:
        prefix_parts.append(f"test_top1_error={test_error:.3f}")
    if best_test_error is not None:
        prefix_parts.append(f"test_top1_error_best={best_test_error:.3f}")
    prefix = "" if not prefix_parts else " ".join(prefix_parts) + " | "
    print("    " + prefix + (" | ".join(summary_parts) if summary_parts else "No tracked metrics found."))


def main():
    if MODE == "run":
        scan_cases = build_scan_cases()
        jobs = [(case, seed) for case in scan_cases for seed in range(1, N_SEEDS + 1)]
        print(f"Running {len(jobs)} jobs with {N_PARALLEL} workers")

        for (case, seed), result, results in run_jobs(jobs, run_one, sort_results, save_results, N_PARALLEL):
            print(f"  scan={case['scan_label']} seed={seed}")
            print_result_summary(result)

        aggregated_results = aggregate_results(results)
        print("\nSummary")
        print(render_metric_tables(aggregated_results, SUMMARY_METRICS))
    elif MODE == "plot":
        plot_apical_pr_summary()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
