# GEMINI.md - gomlx-pgvect-rag

This document defines the project-specific instructions and context for Gemini CLI when working on the `gomlx-pgvect-rag` project.

## Project Overview
A Multimodal Retrieval-Augmented Generation (RAG) system built with:
- **Language/Framework:** Go (Golang 1.23+)
- **ML Engine:** [GoMLX](https://github.com/gomlx/gomlx) (XLA-accelerated)
- **MCP Server** [Official go MCP SDK](https://github.com/modelcontextprotocol/go-sdk)
- **Vector Database:** PostgreSQL with [pgvector](https://github.com/pgvector/pgvector)
- **Models:** T5Gemma 2 (Multimodal Encoder-Decoder)
  - **Foundation:** Based on **Gemma 3** architecture (UL2 adaptation).
  - **Encoder:** SigLIP Vision Encoder (1152-dim) + Gemma 3 Text Encoder (640-dim).
  - **Decoder:** Gemma 3 for generation.
  - **Embedding Dimension:** 640 (based on Gemma 3 text encoder hidden size).

## Engineering Standards
- **Go Conventions:** Standard directory structure (`cmd/`, `internal/`, `pkg/`).
- **GoMLX Graph Operations:** 
  - Image Preprocessing: Resize to **896x896**, normalize to `[-1, 1]` (Mean: 0.5, Std: 0.5).
  - Embedding Extraction: **Mean Pooling** on the encoder's last hidden state across the sequence dimension.
- **Database:** Use `pgx/v5` and `pgvector-go`. 
  - Use **Cosine Distance** (`<=>`) for similarity searches.
  - Implement **UPSERT** logic using `ON CONFLICT` for the `filesys` table.
- **Performance:** Optimize HNSW index with `m = 16`, `ef_construction = 128`, and `ef_search = 100`.

## Key Files & Directories
- `internal/embedder/`: SigLIP normalization and T5Gemma 2 encoder logic.
- `internal/db/`: Postgres schema, HNSW indexing, and BYTEA/JSONB handling.
- `internal/rag/`: Retrieval (vector search) and Generation (Gemma 2 decoder) orchestration.
- `models/`: Weights converted for GoMLX (.bin / .safetensors).

## Documentation
- Always update `SPECIFICATION.md` for architectural changes.
- Track implementation progress in `IMPLEMENTATION_PLAN.md`.
