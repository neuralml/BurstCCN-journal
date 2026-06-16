import json
from pathlib import Path
import sys

import numpy as np

import matplotlib.pyplot as plt

from common import run_command, run_jobs, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "simple_increasing_relative_target_ablation_scan.json"
PDF_PATH = SCRIPT_DIR / "plots" / "simple_increasing_ablation_scan.pdf"

WANDB_MODE = "disabled"
GROUP = "ablation"
RUN_NAME = "simple_increasing_task_burstccn_dales"
N_SEEDS = 6
N_PARALLEL = 10
PLOT_ONLY = False

INTERNEURON_TYPES = ["control", "pv", "vip", "sst", "ndnf"]
INPUT_VALUE_RANGE = [2.0, 2.0]
INITIAL_DIRECT_WEIGHT_RANGE = [0.3, 0.5]
# TARGET_CHANGE_RANGE = [0.2, 0.4]
TARGET_CHANGE_RANGE = [0.1, 0.2]
Y_SCALE_RANGE = [0.5, 1.5]
# Y_SCALE_RANGE = [5.0, 15]

DISPLAY_NAME_BY_TYPE = {
    "control": "control",
    "pv": "PV",
    "vip": "VIP",
    "sst": "SST",
    "ndnf": "NDNF",
}
COLOR_BY_TYPE = {
    "control": "#999999",
    "pv": "#85caa0",
    "vip": "#f58d73",
    "sst": "#7d9ed2",
    "ndnf": "#d6a8cd",
}

RESULT_SCALAR_KEYS = [
    "seed",
    "input_value",
    "initial_direct_weight",
    "target_change",
    "y_scale",
    "initial_output",
    "new_target_value",
    "w_direct_initial",
    "w_direct_final",
    "w_direct_diff",
    "w_initial",
    "w_final",
    "w_diff",
    "event_rate_initial",
    "event_rate_final",
    "event_rate_diff",
    "p_t_initial",
    "p_t_final",
    "p_t_diff",
]


def parse_trigger_output(output_text):
    parsed = {}
    for line in output_text.splitlines():
        line = line.strip()
        if line.startswith("[Trigger] Initial output ="):
            parsed["initial_output"] = float(line.split("=", maxsplit=1)[1])
        elif line.startswith("[Trigger] New target value ="):
            parsed["new_target_value"] = float(line.split("=", maxsplit=1)[1])
        elif line.startswith("[InterneuronAblationTrigger] "):
            key, value = line[len("[InterneuronAblationTrigger] ") :].split("=", maxsplit=1)
            key = key.strip()
            if key == "ablation_type":
                continue
            value = value.strip()
            parsed[key] = float(value)
    return parsed


def run_one(interneuron_type, seed):
    rng = np.random.default_rng(seed)

    input_value = float(rng.uniform(*INPUT_VALUE_RANGE))
    initial_direct_weight = float(rng.uniform(*INITIAL_DIRECT_WEIGHT_RANGE))
    target_change = float(rng.uniform(*TARGET_CHANGE_RANGE))
    y_scale = float(rng.uniform(*Y_SCALE_RANGE))

    cmd = [
        sys.executable,
        "train.py",
        "--config-name=simple_increasing_task_burstccn_dales",
        f"+run_name={RUN_NAME}",
        f"+group={GROUP}",
        f"training.seed={seed}",
        f"dataset.input_value={input_value}",
        f"model.Y.scale={y_scale}",
        "model.apical_bias_learning=true",
        f"training.triggers.interneuron_ablation.interneuron_type={interneuron_type}",
        f"training.triggers.relative_target_change.target_change={target_change}",
        f"training.triggers.weight_override_l0.value={initial_direct_weight}",
        "model.W.optimiser.lr=1.0"
    ]

    returncode, output_text = run_command(cmd, cwd=REPO_ROOT)
    parsed = parse_trigger_output(output_text)

    return {
        "interneuron_type": interneuron_type,
        "seed": seed,
        "returncode": returncode,
        "input_value": input_value,
        "initial_direct_weight": initial_direct_weight,
        "target_change": target_change,
        "y_scale": y_scale,
        "output_tail": "\n".join(output_text.splitlines()[-60:]),
        **parsed,
    }


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "interneuron_types": INTERNEURON_TYPES,
            "input_value_range": INPUT_VALUE_RANGE,
            "initial_direct_weight_range": INITIAL_DIRECT_WEIGHT_RANGE,
            "target_change_range": TARGET_CHANGE_RANGE,
            "y_scale_range": Y_SCALE_RANGE,
            "n_seeds": N_SEEDS,
            "group": GROUP,
            "wandb_mode": WANDB_MODE,
            "n_parallel": N_PARALLEL,
        },
        "interneuron_results": build_interneuron_results(results),
    }
    write_json_atomic(RESULTS_PATH, payload)


def build_interneuron_results(results):
    grouped = {
        interneuron_type: {key: [] for key in RESULT_SCALAR_KEYS}
        for interneuron_type in INTERNEURON_TYPES
    }
    for result in sort_results(results):
        values = grouped[result["interneuron_type"]]
        for key in RESULT_SCALAR_KEYS:
            values[key].append(result.get(key))
    return grouped


def flatten_interneuron_results(interneuron_results):
    results = []
    for interneuron_type in INTERNEURON_TYPES:
        values = interneuron_results.get(interneuron_type, {})
        n_runs = max((len(items) for items in values.values()), default=0)
        for index in range(n_runs):
            result = {"interneuron_type": interneuron_type, "returncode": 0}
            for key in RESULT_SCALAR_KEYS:
                items = values.get(key, [])
                if index < len(items):
                    result[key] = items[index]
            results.append(result)
    return sort_results(results)


def sort_results(results):
    return sorted(
        results,
        key=lambda result: (INTERNEURON_TYPES.index(result["interneuron_type"]), result["seed"]),
    )


def load_results():
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if "interneuron_results" in payload:
        return flatten_interneuron_results(payload["interneuron_results"])
    return sort_results(payload["results"])


def values_by_type(results, key, transform=None):
    grouped = {interneuron_type: [] for interneuron_type in INTERNEURON_TYPES}
    for result in results:
        value = result.get(key)
        if value is None:
            continue
        grouped[result["interneuron_type"]].append(transform(value) if transform else value)
    return grouped


def mean_or_none(values):
    return None if len(values) == 0 else float(np.mean(values))


def render_summary_table(results):
    headers = ["type", "n", "ok", "w_diff", "e_diff", "p_t_init", "p_t_final", "p_t_diff"]
    rows = []
    for interneuron_type in INTERNEURON_TYPES:
        type_results = [result for result in results if result["interneuron_type"] == interneuron_type]
        row_values = {
            "w_diff": mean_or_none([result["w_diff"] for result in type_results if "w_diff" in result]),
            "e_diff": mean_or_none(
                [result["event_rate_diff"] for result in type_results if "event_rate_diff" in result]
            ),
            "p_t_init": mean_or_none([result["p_t_initial"] for result in type_results if "p_t_initial" in result]),
            "p_t_final": mean_or_none([result["p_t_final"] for result in type_results if "p_t_final" in result]),
            "p_t_diff": mean_or_none([result["p_t_diff"] for result in type_results if "p_t_diff" in result]),
        }
        rows.append(
            [
                interneuron_type,
                str(len(type_results)),
                str(sum(result["returncode"] == 0 for result in type_results)),
                *["" if row_values[key] is None else f"{row_values[key]:.4f}" for key in row_values],
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def format_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    return "\n".join(
        [format_row(headers), "-+-".join("-" * width for width in widths)]
        + [format_row(row) for row in rows]
    )


def apply_plot_style():
    import matplotlib as mpl

    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["xtick.major.width"] = 1.5
    mpl.rcParams["ytick.major.width"] = 1.5
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["savefig.dpi"] = 600


def plot_bar(ax, grouped_values, y_label):
    labels = INTERNEURON_TYPES
    x = np.arange(len(labels))
    colours = [COLOR_BY_TYPE[label] for label in labels]
    y_lists = [grouped_values[label] for label in labels]
    means = [np.nan if len(values) == 0 else np.mean(values) for values in y_lists]
    sems = [0 if len(values) <= 1 else np.std(values, ddof=1) / np.sqrt(len(values)) for values in y_lists]

    ax.bar(x, means, 0.5, color=colours, edgecolor="black", linewidth=1)
    ax.errorbar(x, means, yerr=sems, color="black", fmt="none", capsize=3, linewidth=1.5, zorder=6)

    rng = np.random.default_rng(seed=0)
    for index, values in enumerate(y_lists):
        if not values:
            continue
        ax.scatter(
            x[index] + rng.normal(0, 0.05, size=len(values)),
            values,
            color="black",
            s=10,
            zorder=5,
            alpha=0.65,
        )

    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticklabels(
        [DISPLAY_NAME_BY_TYPE[label] for label in labels],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(y_label)


def plot_results(results):
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), squeeze=False)
    axes = axes.reshape(-1)

    plot_bar(axes[0], values_by_type(results, "event_rate_diff"), "Difference in\nevent rate")
    plot_bar(
        axes[1],
        values_by_type(results, "p_t_initial", transform=lambda value: value - 0.5),
        "Difference of initial\n$p_t$ from 0.5",
    )
    plot_bar(axes[2], values_by_type(results, "w_diff"), "Difference in\nsynaptic weight")

    fig.tight_layout()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    print(f"Saved plot to: {PDF_PATH}")
    plt.show()


def main():
    if PLOT_ONLY:
        results = load_results()
    else:
        jobs = [
            (interneuron_type, seed)
            for interneuron_type in INTERNEURON_TYPES
            for seed in range(1, N_SEEDS + 1)
        ]
        results = []
        print(
            f"Running {len(jobs)} jobs with {N_PARALLEL} workers "
            f"input~U({INPUT_VALUE_RANGE[0]:.4f},{INPUT_VALUE_RANGE[1]:.4f}) "
            f"w_direct~U({INITIAL_DIRECT_WEIGHT_RANGE[0]:.4f},{INITIAL_DIRECT_WEIGHT_RANGE[1]:.4f}) "
            f"target_change~U({TARGET_CHANGE_RANGE[0]:.4f},{TARGET_CHANGE_RANGE[1]:.4f}) "
            f"Y_scale~U({Y_SCALE_RANGE[0]:.4f},{Y_SCALE_RANGE[1]:.4f})"
        )

        for (interneuron_type, seed), result, results in run_jobs(
            jobs,
            lambda job: run_one(*job),
            sort_results,
            save_results,
            N_PARALLEL,
        ):
            print(
                f"  type={interneuron_type} seed={seed} "
                f"exit={result['returncode']} "
                f"w_diff={result.get('w_diff', np.nan):.4f} "
                f"e_diff={result.get('event_rate_diff', np.nan):.4f} "
                f"p_t_diff={result.get('p_t_diff', np.nan):.4f}"
            )
            if result["returncode"] != 0:
                raise SystemExit(f"Run failed:\n{result['output_tail']}")

    print(render_summary_table(results))
    plot_results(results)


if __name__ == "__main__":
    main()
