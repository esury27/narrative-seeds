import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RAW_PASSAGES_PATH = PROJECT_ROOT / "data" / "raw_passages.json"
RECONSTRUCTIONS_PATH = PROJECT_ROOT / "data" / "reconstructions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "embeddings.json"

EMBEDDING_MODEL = "text-embedding-3-small"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_embedding(client, text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Check your .env file.")

    client = OpenAI()

    passages = load_json(RAW_PASSAGES_PATH)
    reconstructions = load_jsonl(RECONSTRUCTIONS_PATH)

    output = {
        "embedding_model": EMBEDDING_MODEL,
        "originals": {},
        "reconstructions": []
    }

    total = len(passages) + len(reconstructions)
    count = 0

    for passage in passages:
        count += 1
        print(f"[{count}/{total}] embedding original {passage['id']}")

        output["originals"][passage["id"]] = {
            "title": passage["title"],
            "genre": passage["genre"],
            "embedding": get_embedding(client, passage["text"])
        }

    for i, record in enumerate(reconstructions):
        count += 1
        print(
            f"[{count}/{total}] embedding reconstruction "
            f"{record['passage_id']} | {record['condition']} | {record['max_tokens']}"
        )

        output["reconstructions"].append({
            "index": i,
            "passage_id": record["passage_id"],
            "passage_title": record["passage_title"],
            "genre": record["genre"],
            "condition": record["condition"],
            "max_tokens": record["max_tokens"],
            "sender_model": record["sender_model"],
            "receiver_model": record["receiver_model"],
            "embedding": get_embedding(client, record["reconstruction"])
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"\nDone. Wrote embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()