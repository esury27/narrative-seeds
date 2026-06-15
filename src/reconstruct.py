import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MEMORY_MESSAGES_PATH = PROJECT_ROOT / "data" / "memory_messages.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reconstructions.jsonl"

MODEL = "gpt-4.1-mini"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(memory_message):
    return f"""
You are given a memory message written by another language model.

The original text is hidden from you.

Reconstruct the original text as faithfully as possible using only the memory message.

Preserve factual content, emotional tone, causal structure, narrative perspective, and style where possible.

Output only the reconstructed text.

Memory message:
{memory_message}
""".strip()


def reconstruct_text(client, prompt):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    return response.output_text.strip()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Check your .env file.")

    memory_records = load_jsonl(MEMORY_MESSAGES_PATH)

    OUTPUT_PATH.write_text("", encoding="utf-8")

    client = OpenAI()

    total = len(memory_records)

    for i, record in enumerate(memory_records, start=1):
        print(
            f"[{i}/{total}] "
            f"{record['passage_id']} | "
            f"{record['condition']} | "
            f"{record['max_tokens']} tokens"
        )

        prompt = build_prompt(record["memory_message"])
        reconstruction = reconstruct_text(client, prompt)

        output_record = {
            "passage_id": record["passage_id"],
            "passage_title": record["passage_title"],
            "genre": record["genre"],
            "condition": record["condition"],
            "max_tokens": record["max_tokens"],
            "sender_model": record["sender_model"],
            "receiver_model": MODEL,
            "memory_message": record["memory_message"],
            "reconstruction": reconstruction,
        }

        append_jsonl(OUTPUT_PATH, output_record)

    print(f"\nDone. Wrote reconstructions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()