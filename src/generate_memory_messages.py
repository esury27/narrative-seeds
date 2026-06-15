import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RAW_PASSAGES_PATH = PROJECT_ROOT / "data" / "raw_passages.json"
CONDITIONS_PATH = PROJECT_ROOT / "data" / "conditions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "memory_messages.jsonl"

MODEL = "gpt-4.1-mini"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(passage, max_tokens, instruction):
    return f"""
You will be shown an original text.

Write a memory message of at most {max_tokens} tokens for a future language model.

The future model will not see the original text. It will only see your memory message and will be asked to reconstruct the original text as faithfully as possible.

Instruction condition:
{instruction}

Preserve the information most useful for later reconstruction.

Output only the memory message.

Original text:
{passage["text"]}
""".strip()


def generate_memory_message(client, prompt):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    return response.output_text.strip()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env or export it in your terminal.")

    passages = load_json(RAW_PASSAGES_PATH)
    conditions = load_json(CONDITIONS_PATH)

    OUTPUT_PATH.write_text("", encoding="utf-8")

    client = OpenAI()

    total = len(passages) * len(conditions["lengths"]) * len(conditions["instructions"])
    count = 0

    for passage in passages:
        for max_tokens in conditions["lengths"]:
            for condition in conditions["instructions"]:
                count += 1

                print(
                    f"[{count}/{total}] "
                    f"{passage['id']} | "
                    f"{condition['condition']} | "
                    f"{max_tokens} tokens"
                )

                prompt = build_prompt(
                    passage=passage,
                    max_tokens=max_tokens,
                    instruction=condition["instruction"],
                )

                memory_message = generate_memory_message(client, prompt)

                record = {
                    "passage_id": passage["id"],
                    "passage_title": passage["title"],
                    "genre": passage["genre"],
                    "condition": condition["condition"],
                    "max_tokens": max_tokens,
                    "sender_model": MODEL,
                    "memory_message": memory_message,
                }

                append_jsonl(OUTPUT_PATH, record)

    print(f"\nDone. Wrote memory messages to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()