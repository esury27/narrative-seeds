# Self-Addressed Memory in Language Models

*Measuring compression, reconstruction, and narrative fidelity across context resets*

This project studies how large language models compress narratives into short "memory messages" for a future model instance to reconstruct. A model is given an original passage and asked to write a compressed message addressed to a future instance. That future instance never sees the original — only the memory message — and must reconstruct the passage as faithfully as possible.

The goal isn't to model human memory directly, but to treat LLMs as computational systems for studying externalized memory, lossy compression, and reconstruction under uncertainty.

## How it works

```
original passage
  → self-addressed memory message
  → fresh-context reconstruction
  → fidelity evaluation
```

## Research question

How do large language models compress narratives into self-addressed memory messages, and how do message length and instruction type affect the fidelity of later reconstruction?

(I hypothesize that emotional and stylistic fidelity degrade faster under high compression than concrete factual structures, and that semantic embeddings will insufficiently capture these phenomenological distinctions)

## Current MVP

The MVP runs a full grid over:

- 12 original passages
- 3 memory-message lengths: 50, 100, 200 tokens
- 4 compression instructions: `neutral`, `factual`, `emotional`, `anomaly_preserving`
- 1 reconstruction per memory message
- embedding-based cosine similarity between each reconstruction and its original

That's 12 × 3 × 4 = 144 reconstructions.

## Repository structure

```
identity-ml/
├── data/
│   ├── raw_passages.json          # original corpus (12 passages)
│   ├── conditions.json            # lengths + instruction conditions
│   ├── memory_messages.jsonl      # generated memory messages
│   ├── reconstructions.jsonl      # reconstructions from memory messages
│   ├── embeddings.json            # embeddings for originals + reconstructions
│   └── results/
│       ├── similarity_scores.csv
│       ├── summary_by_condition.csv
│       ├── summary_by_length.csv
│       ├── summary_by_genre.csv
│       └── initial_findings.md    # full result tables and breakdowns
├── src/
│   ├── generate_memory_messages.py
│   ├── reconstruct.py
│   ├── embed.py
│   ├── similarity.py
│   └── analyze_results.py
├── .env
└── README.md
```

## Data format

Each passage in `raw_passages.json`:

```json
{
  "id": "narrative_001",
  "title": "...",
  "genre": "...",
  "source_type": "...",
  "text": "..."
}
```

Experimental conditions in `conditions.json`:

```json
{
  "lengths": [50, 100, 200],
  "instructions": [
    { "condition": "neutral", "instruction": "Preserve the information most useful for faithful reconstruction." },
    { "condition": "factual", "instruction": "Prioritize concrete facts, events, names, objects, places, and sequence." },
    { "condition": "emotional", "instruction": "Prioritize emotional tone, atmosphere, affective arc, and inner experience." },
    { "condition": "anomaly_preserving", "instruction": "Prioritize strange, unexpected, distinctive, or schema-violating details." }
  ]
}
```

## Setup

Create a `.env` file in the project root (do not commit it):

```
OPENAI_API_KEY=your_real_api_key_here
```

## Running the pipeline

Run the scripts in order:

```bash
python src/generate_memory_messages.py   # passages + conditions → memory_messages.jsonl
python src/reconstruct.py                 # memory_messages.jsonl → reconstructions.jsonl
python src/embed.py                        # originals + reconstructions → embeddings.json
python src/similarity.py                   # embeddings.json → similarity_scores.csv
python src/analyze_results.py              # similarity_scores.csv → summaries by condition, length, genre
```

## Results

Initial findings from the MVP run:

- **Longer memory messages reconstruct better.** Fidelity rises with message length, with the largest jump between 50 and 100 tokens and a smaller gain from 100 to 200.
- **Factual and neutral compression score highest** by embedding similarity. Emotional compression scores lowest — likely because it preserves tone and atmosphere at the expense of concrete semantic content, not necessarily because it's worse overall.
- **Concrete, object-rich passages reconstruct most reliably.** Abstract, anomalous, philosophical, or stream-of-consciousness passages are harder to reconstruct faithfully from a compressed message.

A useful working model for these results: *reconstruction = preserved information + model prior + decoding strategy*. When a passage offers concrete objects and familiar structure, the model has more stable cues to rebuild from.

Full tables and per-condition breakdowns are in [`data/results/initial_findings.md`](data/results/initial_findings.md).

## Limitations

Embedding similarity is the only evaluation metric in the MVP. It captures broad semantic overlap but misses stylistic similarity, emotional fidelity, exact phrasing, distinctive details, narrative voice, causal structure, and hallucinated additions. Results should be treated as preliminary until feature-based evaluation and pairwise LLM judging are added.

## Next steps

- Add plots for similarity by length, condition, and genre.
- Extract features for characters, events, setting, emotional trajectory, distinctive details, and style.
- Add pairwise LLM judging across factual, emotional, causal, stylistic, and overall fidelity, then fit a Bradley-Terry model.
- Generate multiple reconstructions per memory message to measure reconstruction variance.
- Compare same-model vs. cross-model sender-receiver pairs.
