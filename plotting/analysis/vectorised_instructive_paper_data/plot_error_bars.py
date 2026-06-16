from pathlib import Path
import csv

import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent
P_PLUS_COLOUR = "#d62728"
P_MINUS_COLOUR = "#1f77b4"


def load_bar_data(csv_name: str):
    csv_path = DATA_DIR / csv_name
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 4:
        raise ValueError(f"Expected at least 4 rows in {csv_name}, got {len(rows)}")

    mean_row = rows[2]
    mean_plus_err_row = rows[3]

    p_plus_mean = float(mean_row[1])
    p_plus_mean_plus_err = float(mean_plus_err_row[1])
    p_minus_mean = float(mean_row[3])
    p_minus_mean_plus_err = float(mean_plus_err_row[3])

    means = [p_plus_mean, p_minus_mean]
    errors = [
        abs(p_plus_mean_plus_err - p_plus_mean),
        abs(p_minus_mean_plus_err - p_minus_mean),
    ]
    labels = ["P+", "P-"]
    colours = [P_PLUS_COLOUR, P_MINUS_COLOUR]

    return labels, means, errors, colours


def plot_bar_csv(csv_name: str, title: str):
    labels, means, errors, colours = load_bar_data(csv_name)

    fig, ax = plt.subplots(figsize=(2.5, 4.0))
    x = range(1, len(labels)+1)

    ax.bar(
        x,
        means,
        yerr=errors,
        color=colours,
        width=0.5,
        edgecolor="black",
        linewidth=1.0,
        capsize=4,
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Signal")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim([0.5, 2.5])
    ax.set_ylim([-0.2, 0.1])

    fig.tight_layout()

    return fig, ax


def main():
    plot_bar_csv("decreasing_error_bar.csv", "Decreasing error")
    plot_bar_csv("increasing_error_bar.csv", "Increasing error")
    plot_bar_csv("decreasing_error_bar_with_ndnf.csv", "Decreasing error")
    plot_bar_csv("increasing_error_bar_with_ndnf.csv", "Increasing error")
    plt.show()


if __name__ == "__main__":
    main()
