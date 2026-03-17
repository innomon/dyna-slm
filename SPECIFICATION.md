# SPECIFICATION.md - Multimodal RAG with T5Gemma 2 (Gemma 3 Architecture)

This document details the architecture for the T5Gemma 2 RAG system, based on the **Gemma 3** foundation.

## 1. Technical Stack
- **ML Framework:** GoMLX (XLA-accelerated model inference).
- **Models:**
  - **T5Gemma 2-270M** (based on Gemma 3 architecture).
    - Vision Encoder: SigLIP (896x896 input, 256 tokens).
    - Text Encoder: Gemma 3 (UL2 adaptation).
    - Generative Decoder: Gemma 3.
  - **IBM Granite 4.0 350M-H** (Hybrid SSM/Transformer).
    - Architecture: Hybrid Mamba-2 + Transformer (GQA).
    - Embedding Dimension: 768.
- **Database:** PostgreSQL + `pgvector`.
- **Drivers:** `pgx/v5`, `pgvector-go`.

## 2. Multimodal Embedding Strategy
- **Image Preprocessing:**
  - Resize to **896x896** using Lanczos resampling.
  - Normalize to `[-1, 1]` within the GoMLX graph: `(input/255.0 - 0.5) / 0.5`.
- **Tokenization:**
  - Image: 256 visual tokens (SigLIP).
  - Text: Gemma 3 tokenizer (256k vocab).
- **Encoding:** 
  - Vision tokens (1152-dim) are projected to the text encoder space.
  - Concatenate visual and text tokens -> Gemma 3 Encoder blocks.
- **Pooling:** Perform **Mean Pooling** across the sequence dimension of the **last hidden state** (640-dim for the 270M model).
- **Output:** Vector of dimension **640**.

## 3. Database Schema (PostgreSQL)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(640) -- Support for T5Gemma 2 (640-dim) or Granite 4.0 350M-H (768-dim)
);

-- Index metadata
CREATE INDEX IF NOT EXISTS idx_filesys_metadata ON filesys USING GIN (metadata);

-- Index embeddings with HNSW for Cosine Similarity
CREATE INDEX IF NOT EXISTS idx_filesys_embedding ON filesys 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);
```

## 4. MCP Server Architecture
The system is packaged as an **MCP Server**, exposing tools for ingestion and search.

### 4.1. Tools
- **`search_multimodal`**: Search using `query_text` or `query_image_path`.
- **`ingest_asset`**: Ingest a local file, generate a 640-dim embedding, and UPSERT into PG.
- **`get_asset_details`**: Retrieve full metadata/content for a specific path.
- **`db_migrate`**: (CLI Utility) Re-embed assets for architecture migration and metadata updates.

## 5. Deployment Considerations
- **Attention:** Implements alternating sliding window (512 tokens) and full attention (every 6th layer).
- **Memory:** Requires sufficient VRAM/RAM for GoMLX XLA backend and pgvector HNSW index.
- **Distance Metric:** Always use **Cosine Distance** (`<=>`).

## 6. OpenAI-Compatible API Layer
The system provides a JWT-secured REST API compatible with the OpenAI specification for seamless integration with existing tools (e.g., AnythingLLM, Goose).

### 6.1. Authentication
- **Algorithm:** EdDSA (Ed25519) asymmetric signatures.
- **Requirement:** `Authorization: Bearer <token>` header for all protected endpoints.
- **Key Management:** A stable Ed25519 keypair is derived from the `JWT_SECRET` seed (using SHA-256).

### 6.2. Endpoints
- **`POST /v1/chat/completions`**: RAG-augmented chat interface.
  - Automatically performs similarity search on user input.
  - Augments the prompt with retrieved context.
  - **Function Calling**: Supports OpenAI-compatible `tools` and `tool_choice`.
- **`POST /v1/responses`**: Unified, stateful-ready interface (successor to Chat Completions).
  - Uses the "Items" model for flexible, agentic workflows.
  - Supports `instructions` and `input` (text or item array).
  - **Function Calling**: Supports `tools`, `tool_choice`, and returns `function_call` / `function_call_output` items.
- **`POST /v1/embeddings`**: Direct access to the T5Gemma 2 multimodal embedding model.
  - Accepts text input and returns 640-dimension vectors.
- **`GET /v1/models`**: Returns information about the available Dyna-SLM models.
- **`GET /v1/tools`**: Returns the list of available tools for function calling.
- **`GET /auth/token`**: (Development Only) Utility to generate a temporary test token.
### 6.4. Available Tools (Function Calling)
The API exposes the following tools, matching the MCP server capabilities:
- **`search_multimodal`**: Search for relevant text or image assets in the multimodal RAG store.
  - Arguments: `model` (required), `query_text`, `query_image_path`, `limit`.
- **`ingest_asset`**: Ingest a new file (image or text) into the multimodal RAG store.
  - Arguments: `model` (required), `path` (required).
- **`list_variants`**: List all available Dyna-SLM model variants.

