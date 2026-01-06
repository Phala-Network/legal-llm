# Law Assistant

A fine-tuned legal assistant powered by **Qwen 2.5 72B** (via Unsloth), **Agentic RAG**, and an **OpenAI-compatible API**.

## Overview

This project implements a high-performance legal AI assistant. It uses "Agentic RAG," where the model is fine-tuned to autonomously generate search queries (`<search>query</search>`) when triggered by a user's question, retrieves relevant case law, and synthesizes a citation-backed answer.

### Key Features

- **Agentic RAG**: Model autonomously decides when and what to search.
- **Hybrid Retrieval**: Combines **Tantivy** (Full-Text Search) for global routing and **ChromaDB** (Vector Search) for deep context retrieval.
- **Neighborhood-Based Sharding**: Effectively sharded vector DB supporting 6.9M+ cases by grouping them into citation neighborhoods (Ego-Networks).
- **Efficient Fine-tuning**: Uses **Unsloth** for 4-bit LoRA training of Llama-3.3 and Qwen-2.5.
- **High-Scale Data Gen**: Includes a multi-stage **OpenAI Batch API** pipeline for generating massive synthetic datasets at 50% lower cost.
- **RAG Pipeline**: Semantic ingestion using **ChromaDB** and **OpenAI Embeddings**.
- **OpenAI-Compatible Server**: FastAPI-based server that supports streaming, chat completions, and a built-in legal case viewer.

## Project Structure

- `src/rag`: Ingestion (`ingest.py`) and retrieval logic.
- `src/data_gen`: Synthetic data generation.
  - `generate.py`: Direct generation (low latency).
  - `generate_batch.py`: Automated multi-stage pipeline using OpenAI Batch API.
- `src/finetune`: Unsloth training scripts and data sampling utilities.
- `src/inference`:
  - `server.py`: OpenAI-compatible FastAPI server.
  - `chat.py`: CLI-based interactive chat.
- `data/`: Directory for raw case law JSON files.

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone <repo_url>
cd law-assistant

# Install dependencies
uv sync
```

**Environment Setup (`.env`)**:

```env
OPENAI_API_KEY=your_key
EMBEDDING_MODEL_NAME=text-embedding-3-large
```

## Usage

### 1. Download Data

You can download case law data using the provided scripts.

**Download All Reporters (Recommended)**:
This script downloads metadata and volumes for all available reporters. Supports parallel downloading.

```bash
# Download all reporters with 4 workers and 0.5s delay
uv run src/scripts/download_all_reporters.py --output_dir data --workers 4 --delay 0.5
```

**Download Single Reporter**:
If you only need a specific reporter.

```bash
uv run src/scripts/download_reporter.py --reporter cal-rptr-3d --max_volumes 5
```

### 2. Sharding & Neighborhood Analysis

For large datasets (6.9M+), analyze citation neighborhoods to optimize vector DB performance.

```bash
uv run src/scripts/analyze_case_neighborhoods.py --base_dir data --search_dir data --output data/case_neighborhoods.json
```

### 3. Unified Ingestion (Sharding -> Vector DB -> Router Index)

Place your case law JSONs in `data/` and run the unified ingester. This script performs three critical steps:

1. **Generates Shard Assignments** based on citation neighborhoods.
2. **Ingests semi-structured case text** into sharded ChromaDB collections.
3. **Builds a Global Router Index** (Tantivy) for full-text routing.

```bash
uv run src/rag/ingest.py --neighborhoods data/case_neighborhoods.json
```

**Tip: Testing on a subset**:
To verify the pipeline before a full run, use `--search_dir` to only ingest a specific reporter or volume:

```bash
uv run src/rag/ingest.py --search_dir data/us/1 --neighborhoods data/case_neighborhoods.json
```

### 4. Integrated Retrieval (End-to-End)

The `CaseRetriever` now handles the entire two-stage retrieval process internally, leveraging the global router index.

**End-to-End Search**:

```bash
uv run src/rag/retriever.py "locomotive brakes livestock"
```

**Focused Search (Neighborhood Aware)**:
If you know the specific case context, you can focus the search on its citation neighborhood shard:

```bash
uv run src/rag/retriever.py "your legal question" --focus "us/1/0001-01"
```

**How it works**:

1. **Global Router**: Queries the Tantivy index to find the best candidate "Anchor Case".
2. **Shard Resolution**: Uses `ShardManager` to identify the shard containing that case's citation neighborhood.
3. **Vector Retrieval**: Performs semantic search within the identified ChromaDB shard(s).
4. **Reranking**: Scores and ranks the combined candidates using a Cross-Encoder.

### 5. Generate Synthetic Data

Choose between direct generation or the batch pipeline (recommended for large scale).

**Direct Generation**:

```bash
uv run src/data_gen/generate.py --num_samples 50
```

**Batch Pipeline**:

```bash
uv run src/data_gen/generate_batch.py --pipeline --num_samples 1000
```

### 6. Fine-tune Model

Train the model using the generated `training_data.jsonl`.

```bash
uv run src/finetune/train.py
```

### 7. Run Inference

**Option A: Interactive CLI**

```bash
uv run src/inference/chat.py
```

**Option B: API Server**
Starts an OpenAI-compatible server on port 8000.

```bash
uv run src/inference/server.py
```

- **Endpoint**: `POST /v1/chat/completions`
- **Case Viewer**: `GET /cases/{case_id}` (renders case text as HTML)

## Performance & Sharded Architecture

To handle 6.9M cases efficiently, the system uses a **Shard-Aware Retrieval** architecture. Instead of a single giant index, cases are distributed into **overlapping citation neighborhoods**.

### Dataset Connectivity (6.9M Cases)

Analysis of the full dataset reveals highly dense citation inter-connectivity:

| Metric                        | Value     |
| :---------------------------- | :-------- |
| **Total Cases**               | 6,902,269 |
| **Max Neighborhood Size**     | 69,148    |
| **Average Neighborhood Size** | 19.82     |

**Neighborhood Size Distribution**:

- **Size 1 (Isolated)**: 13.6%
- **Size 2-50**: 78.1%
- **Size 51-500**: 8.1%
- **Size 501+**: 0.07%

For more details on the sharding strategy, see [docs/case_grouping_report.md](docs/case_grouping_report.md).

## Details

- **Base Model**: `unsloth/Qwen2.5-72B-Instruct-bnb-4bit`
- **Global Index**: Tantivy (BM25, Tiered Routing)
- **Vector DB**: ChromaDB (Sharded, 100+ Collections)
- **Embeddings**: `openai/text-embedding-3-large`
