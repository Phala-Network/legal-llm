# Law Assistant

A fine-tuned legal assistant powered by **Qwen 2.5 72B** (via Unsloth), **Agentic RAG**, and an **OpenAI-compatible API**.

## Overview

This project implements a high-performance legal AI assistant. It uses "Agentic RAG," where the model is fine-tuned to autonomously generate search queries (`<search>query</search>`) when triggered by a user's question, retrieves relevant case law, and synthesizes a citation-backed answer.

### Key Features

- **Agentic RAG**: Model autonomously decides when and what to search.
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

### 2. Ingest Data

Place your case law JSONs in `data/` (or wherever you downloaded them) and run the ingester.

```bash
uv run src/rag/ingest.py
```

### 3. Utility Scripts

#### Normalize Citations

If your training data contains file paths instead of IDs, use this to migrate them to unique case IDs.

```bash
uv run src/scripts/migrate_data_to_id.py --input training_data.jsonl --output training_data_migrated.jsonl
```

### 4. Generate Synthetic Data

Choose between direct generation or the batch pipeline (recommended for large scale).

**Direct Generation**:

```bash
uv run src/data_gen/generate.py --num_samples 50
```

**Batch Pipeline**:

```bash
uv run src/data_gen/generate_batch.py --pipeline --num_samples 1000
```

### 5. Fine-tune Model

Train the model using the generated `training_data.jsonl`.

```bash
uv run src/finetune/train.py
```

### 6. Run Inference

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

## Details

- **Base Model**: `unsloth/Qwen2.5-72B-Instruct-bnb-4bit`
- **Vector DB**: ChromaDB (persisted in `chroma_db/`)
- **Embeddings**: `openai/text-embedding-3-large`
