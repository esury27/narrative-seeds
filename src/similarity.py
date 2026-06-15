import json
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "similarity_scores.csv"


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    originals = data["originals"]
    reconstructions = data["reconstructions"]

    rows = []

    for record in reconstructions:
        passage_id = record["passage_id"]

        original_embedding = originals[passage_id]["embedding"]
        reconstruction_embedding = record["embedding"]

        similarity = cosine_similarity(original_embedding, reconstruction_embedding)

        rows.append({
            "passage_id": passage_id,
            "passage_title": record["passage_title"],
            "genre": record["genre"],
            "condition": record["condition"],
            "max_tokens": record["max_tokens"],
            "sender_model": record["sender_model"],
            "receiver_model": record["receiver_model"],
            "cosine_similarity": round(similarity, 4)
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "passage_id",
                "passage_title",
                "genre",
                "condition",
                "max_tokens",
                "sender_model",
                "receiver_model",
                "cosine_similarity"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote similarity scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()