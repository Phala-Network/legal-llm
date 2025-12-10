# Law Assistant

A fine-tuned legal assistant powered by **Qwen 2.5**, **Unsloth**, and **Agentic RAG**.

## Overview

This project implements a legal AI assistant capable of retrieving case law and answering legal questions with citation-backed accuracy. It uses "Agentic RAG," meaning the model is fine-tuned to explicitly call for a search mechanism (`<search>query</search>`) when it needs external information.

### Key Features
- **Agentic RAG**: Model autonomously generates search queries during generation.
- **RAG Pipeline**: Ingestion and semantic search using ChromaDB and `all-MiniLM-L6-v2`.
- **Flexible Fine-tuning**: Uses Unsloth for efficient LoRA fine-tuning of Qwen models.
- **Synthetic Data**: Includes tools to generate high-quality, multi-turn training data from raw case text using OpenAI.

## Project Structure

- `src/rag`: Data ingestion and retrieval logic.
- `src/data_gen`: Data generation scripts for fine-tuning.
- `src/finetune`: Unsloth training scripts.
- `src/inference`: End-to-end inference script with tool-use loop.
- `cases/`: Raw case law data (JSON/HTML).

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone <repo_url>
cd law-assistant

# Install dependencies
uv sync
```

**Requirements**:
- Linux environment (recommended)
- GPU with at least 12GB VRAM (for Qwen-7B 4-bit fine-tuning)
- `OPENAI_API_KEY` (for synthetic data generation)

## Usage

### 1. Ingest Data
Parse raw case files and build the vector database.

```bash
uv run src/rag/ingest.py
```

### 2. Generate Synthetic Data
Create a fine-tuning dataset formatted for Agentic RAG (Question -> Search -> Context -> Answer).

```bash
# Ensure OPENAI_API_KEY is set in .env
uv run src/data_gen/generate.py --num_samples 50
```

### 3. Fine-tune Model
Train the Qwen model using Unsloth on the generated data.

```bash
uv run src/finetune/train.py
```

### 4. Run Inference
Chat with the fine-tuned assistant. It will search for cases if needed.

```bash
uv run src/inference/chat.py --query "What is the holding in the Araujo case?"
```

## Details

- **Model**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Vector DB**: ChromaDB (persisted in `chroma_db/`)
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`
