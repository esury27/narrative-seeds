# Self-Addressed Memory in Language Models

## Measuring Compression, Reconstruction, and Narrative Fidelity Across Context Resets

This project studies how large language models compress narratives into short “memory messages” for future reconstruction. The basic idea is to give an LLM an original passage and ask it to write a compressed message for a future model instance. The future model does not see the original passage; it only sees the memory message and must reconstruct the original as faithfully as possible.

The pipeline is:

```text
original passage
→ self-addressed memory message
→ fresh-context reconstruction
→ fidelity evaluation
```

Rather than treating LLMs as direct models of human memory, this project treats them as computational systems for studying externalized memory, lossy compression, and reconstruction under uncertainty.

---

## Research Question

How do large language models compress narratives into self-addressed memory messages, and how do message length and instruction type affect the fidelity of later reconstruction?

---

## Current MVP

The current MVP uses:

* 12 original passages
* 3 memory-message lengths: 50, 100, and 200 tokens
* 4 compression instruction conditions:

  * neutral
  * factual
  * emotional
  * anomaly-preserving
* 1 reconstruction per memory message
* embedding-based cosine similarity between each reconstruction and its original passage

This produces:

```text
12 passages × 3 lengths × 4 conditions = 144 reconstructions
```

---

## Repository Structure

```text
identity-ml/
├── data/
│   ├── raw_passages.json
│   ├── conditions.json
│   ├── memory_messages.jsonl
│   ├── reconstructions.jsonl
│   ├── embeddings.json
│   └── results/
│       ├── similarity_scores.csv
│       ├── summary_by_condition.csv
│       ├── summary_by_length.csv
│       ├── summary_by_genre.csv
│       └── initial_findings.md
│
├── src/
│   ├── generate_memory_messages.py
│   ├── reconstruct.py
│   ├── embed.py
│   ├── similarity.py
│   └── analyze_results.py
│
├── .env
└── README.md
```

---

## Data Files

### `data/raw_passages.json`

Contains the original corpus of 12 passages.

Each passage has the following structure:

```json
{
  "id": "narrative_001",
  "title": "...",
  "genre": "...",
  "source_type": "...",
  "text": "..."
}
```

### `data/conditions.json`

Defines the experimental conditions.

```json
{
  "lengths": [50, 100, 200],
  "instructions": [
    {
      "condition": "neutral",
      "instruction": "Preserve the information most useful for faithful reconstruction."
    },
    {
      "condition": "factual",
      "instruction": "Prioritize concrete facts, events, names, objects, places, and sequence."
    },
    {
      "condition": "emotional",
      "instruction": "Prioritize emotional tone, atmosphere, affective arc, and inner experience."
    },
    {
      "condition": "anomaly_preserving",
      "instruction": "Prioritize strange, unexpected, distinctive, or schema-violating details."
    }
  ]
}
```

### `data/memory_messages.jsonl`

Contains one generated memory message per passage, length, and instruction condition.

Each line is one JSON object.

### `data/reconstructions.jsonl`

Contains one reconstructed text for each memory message.

Each line is one JSON object.

### `data/embeddings.json`

Contains embeddings for the original passages and the reconstructed passages.

### `data/results/similarity_scores.csv`

Contains cosine similarity scores comparing each reconstruction to its original passage.

### `data/results/summary_by_condition.csv`

Contains average similarity scores grouped by compression instruction.

### `data/results/summary_by_length.csv`

Contains average similarity scores grouped by memory-message length.

### `data/results/summary_by_genre.csv`

Contains average similarity scores grouped by passage genre.

---

## Scripts

### 1. Generate memory messages

```bash
python src/generate_memory_messages.py
```

Reads:

```text
data/raw_passages.json
data/conditions.json
```

Writes:

```text
data/memory_messages.jsonl
```

This script asks the model to compress each original passage into a memory message under each experimental condition.

---

### 2. Generate reconstructions

```bash
python src/reconstruct.py
```

Reads:

```text
data/memory_messages.jsonl
```

Writes:

```text
data/reconstructions.jsonl
```

This script gives each memory message to a fresh model context and asks it to reconstruct the original passage as faithfully as possible.

---

### 3. Generate embeddings

```bash
python src/embed.py
```

Reads:

```text
data/raw_passages.json
data/reconstructions.jsonl
```

Writes:

```text
data/embeddings.json
```

This script embeds both the original passages and the reconstructed passages.

---

### 4. Compute similarity scores

```bash
python src/similarity.py
```

Reads:

```text
data/embeddings.json
```

Writes:

```text
data/results/similarity_scores.csv
```

This script computes cosine similarity between each original passage and its corresponding reconstruction.

---

### 5. Analyze results

```bash
python src/analyze_results.py
```

Reads:

```text
data/results/similarity_scores.csv
```

Writes:

```text
data/results/summary_by_condition.csv
data/results/summary_by_length.csv
data/results/summary_by_genre.csv
```

This script summarizes reconstruction fidelity by condition, memory-message length, and genre.

---

## How to Run the MVP

Run the scripts in this order:

```bash
python src/generate_memory_messages.py
python src/reconstruct.py
python src/embed.py
python src/similarity.py
python src/analyze_results.py
```

---

## Initial Results

### 1. Reconstruction fidelity improved with memory-message length

Mean cosine similarity by memory-message length:

| Max tokens | Mean cosine similarity |
| ---------: | ---------------------: |
|         50 |                 0.7737 |
|        100 |                 0.8411 |
|        200 |                 0.8798 |

This suggests that longer memory messages preserve more reconstructively useful information. The largest improvement occurred between 50 and 100 tokens, while the improvement from 100 to 200 tokens was smaller.

---

### 2. Factual and neutral compression performed best by embedding similarity

Mean cosine similarity by instruction condition:

| Condition          | Mean cosine similarity |
| ------------------ | ---------------------: |
| factual            |                 0.8543 |
| neutral            |                 0.8500 |
| anomaly_preserving |                 0.8455 |
| emotional          |                 0.7762 |

Factual and neutral compression produced the highest similarity scores. Emotional compression produced the lowest score by embedding similarity, possibly because it preserved atmosphere and affect while losing concrete semantic content.

This does not necessarily mean emotional compression is worse overall. It may preserve emotional tone better, but this MVP does not yet directly measure emotional fidelity.

---

### 3. Concrete and image-based passages reconstructed more reliably

The highest-scoring genres were:

| Genre                   | Mean cosine similarity |
| ----------------------- | ---------------------: |
| ordinary_transformation |                 0.8934 |
| uncanny_descriptive     |                 0.8932 |
| childhood_memory        |                 0.8795 |
| grief_medical           |                 0.8752 |

The lowest-scoring genres were:

| Genre                             | Mean cosine similarity |
| --------------------------------- | ---------------------: |
| near_death_anomalous_memory       |                 0.7691 |
| self_observer_projection          |                 0.7728 |
| stream_of_consciousness_spiritual |                 0.7846 |

This suggests that concrete, object-rich, and narratively familiar passages may be easier for the model to reconstruct from compressed memory messages. Abstract, anomalous, philosophical, or stream-of-consciousness passages appear harder to reconstruct faithfully.

---

## Preliminary Interpretation

The results support the idea that LLM reconstruction depends on both preserved information and model priors.

A useful working model is:

```text
Reconstruction = preserved information + model prior + decoding strategy
```

When a passage contains concrete objects, familiar narrative structure, or clear emotional movement, the model can reconstruct it more stably. When a passage is abstract, spiritually intense, anomalous, or structurally unstable, the model has fewer conventional cues to rely on.

The results also suggest that the kind of compression instruction matters. Factual and neutral prompts preserve more broad semantic content, while emotional prompts may produce reconstructions that are affectively similar but less semantically close by embedding similarity.

---

## Limitations

This MVP uses embedding similarity as the main evaluation metric.

Embedding similarity captures broad semantic overlap, but it may miss:

* stylistic similarity
* emotional fidelity
* exact phrasing
* unusual details
* narrative voice
* causal structure
* hallucinated additions

Because of this, the current results should be treated as preliminary. A more complete version should include feature-based evaluation and pairwise LLM judging.

---

## Next Steps

Possible next steps:

1. Add plots for similarity by length, condition, and genre.
2. Add feature extraction for characters, events, setting, emotional trajectory, distinctive details, and style.
3. Add pairwise LLM judging across factual, emotional, causal, stylistic, and overall fidelity.
4. Fit a Bradley-Terry model to pairwise judgments.
5. Generate multiple reconstructions per memory message to measure reconstruction variance.
6. Compare same-model and cross-model sender-receiver pairs.

---

## Environment Setup

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_real_api_key_here
```

Do not commit `.env` to GitHub.

---

## Current Status

Completed:

* corpus creation
* condition setup
* memory-message generation
* reconstruction generation
* embedding generation
* cosine similarity scoring
* summary analysis

Current MVP output is available in:

```text
data/results/
```
