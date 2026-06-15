import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

SUMMARY_BY_LENGTH_PATH = RESULTS_DIR / "summary_by_length.csv"
SUMMARY_BY_CONDITION_PATH = RESULTS_DIR / "summary_by_condition.csv"
SUMMARY_BY_GENRE_PATH = RESULTS_DIR / "summary_by_genre.csv"

PLOTS_DIR = RESULTS_DIR / "plots"


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_bar_plot(rows, x_key, y_key, title, xlabel, ylabel, output_path, rotate_labels=False):
    labels = [row[x_key] for row in rows]
    values = [float(row[y_key]) for row in rows]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    by_length = load_csv(SUMMARY_BY_LENGTH_PATH)
    by_condition = load_csv(SUMMARY_BY_CONDITION_PATH)
    by_genre = load_csv(SUMMARY_BY_GENRE_PATH)

    # Sort lengths numerically
    by_length = sorted(by_length, key=lambda row: int(row["max_tokens"]))

    # Sort genres by score from highest to lowest
    by_genre = sorted(
        by_genre,
        key=lambda row: float(row["mean_cosine_similarity"]),
        reverse=True
    )

    save_bar_plot(
        rows=by_length,
        x_key="max_tokens",
        y_key="mean_cosine_similarity",
        title="Reconstruction Similarity by Memory-Message Length",
        xlabel="Memory-message length",
        ylabel="Mean cosine similarity",
        output_path=PLOTS_DIR / "similarity_by_length.png"
    )

    save_bar_plot(
        rows=by_condition,
        x_key="condition",
        y_key="mean_cosine_similarity",
        title="Reconstruction Similarity by Compression Condition",
        xlabel="Compression condition",
        ylabel="Mean cosine similarity",
        output_path=PLOTS_DIR / "similarity_by_condition.png",
        rotate_labels=True
    )

    save_bar_plot(
        rows=by_genre,
        x_key="genre",
        y_key="mean_cosine_similarity",
        title="Reconstruction Similarity by Genre",
        xlabel="Genre",
        ylabel="Mean cosine similarity",
        output_path=PLOTS_DIR / "similarity_by_genre.png",
        rotate_labels=True
    )

    print(f"Done. Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()