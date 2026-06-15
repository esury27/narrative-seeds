import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "results" / "similarity_scores.csv"

SUMMARY_BY_CONDITION_PATH = PROJECT_ROOT / "data" / "results" / "summary_by_condition.csv"
SUMMARY_BY_LENGTH_PATH = PROJECT_ROOT / "data" / "results" / "summary_by_length.csv"
SUMMARY_BY_GENRE_PATH = PROJECT_ROOT / "data" / "results" / "summary_by_genre.csv"


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows, group_key):
    groups = defaultdict(list)

    for row in rows:
        key = row[group_key]
        score = float(row["cosine_similarity"])
        groups[key].append(score)

    summary_rows = []

    for key, scores in sorted(groups.items()):
        summary_rows.append({
            group_key: key,
            "n": len(scores),
            "mean_cosine_similarity": round(mean(scores), 4),
            "min_cosine_similarity": round(min(scores), 4),
            "max_cosine_similarity": round(max(scores), 4)
        })

    return summary_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(title, rows, key):
    print(f"\n{title}")
    print("-" * len(title))

    for row in rows:
        print(
            f"{row[key]} | "
            f"n={row['n']} | "
            f"mean={row['mean_cosine_similarity']} | "
            f"min={row['min_cosine_similarity']} | "
            f"max={row['max_cosine_similarity']}"
        )


def main():
    rows = load_rows(INPUT_PATH)

    by_condition = summarize(rows, "condition")
    by_length = summarize(rows, "max_tokens")
    by_genre = summarize(rows, "genre")

    write_csv(
        SUMMARY_BY_CONDITION_PATH,
        by_condition,
        ["condition", "n", "mean_cosine_similarity", "min_cosine_similarity", "max_cosine_similarity"]
    )

    write_csv(
        SUMMARY_BY_LENGTH_PATH,
        by_length,
        ["max_tokens", "n", "mean_cosine_similarity", "min_cosine_similarity", "max_cosine_similarity"]
    )

    write_csv(
        SUMMARY_BY_GENRE_PATH,
        by_genre,
        ["genre", "n", "mean_cosine_similarity", "min_cosine_similarity", "max_cosine_similarity"]
    )

    print_summary("Summary by condition", by_condition, "condition")
    print_summary("Summary by length", by_length, "max_tokens")
    print_summary("Summary by genre", by_genre, "genre")

    print("\nDone. Wrote summary CSV files to data/results/")


if __name__ == "__main__":
    main()