# IMPLEMENTATION_PLAN.md - Multimodal RAG with T5Gemma 2

This plan outlines the steps for building the T5Gemma 2 multimodal RAG system.

## Implementation Checklist

### Phase 1: Environment & Project Setup
- [x] Install Go (1.23+), PostgreSQL, and `pgvector` extension.
- [x] Install `libxla` and configure `LD_LIBRARY_PATH`.
- [x] Initialize Go project (`go mod init`).
- [x] Add dependencies:
  - `go get github.com/gomlx/gomlx@latest`
  - `go get github.com/jackc/pgx/v5`
  - `go get github.com/pgvector/pgvector-go`
  - `go get github.com/disintegration/imaging`
  - `go get github.com/modelcontextprotocol/go-sdk/mcp`

### Phase 2: Database Schema (PostgreSQL)
- [x] Create `filesys` table with `path` (PK), `metadata` (JSONB), `content` (BYTEA), `embedding` (vector(640)).
- [x] Create GIN index for metadata.
- [x] Create HNSW index for embeddings with `vector_cosine_ops`.
- [x] Implement `pkg/db/upsert.go` with `ON CONFLICT` support for the `filesys` table.

### Phase 3: Image Preprocessing (GoMLX)
- [x] Implement Go logic to decode JPG/PNG and resize to **896x896** using Lanczos resampling.
- [x] Implement GoMLX graph for SigLIP normalization: `(float32(pixels)/255.0 - 0.5) / 0.5`.
- [x] Verify 4D tensor output shape: `[batch, 896, 896, 3]`.

### Phase 4: Model Integration (T5Gemma 2 Encoder)
- [x] Convert T5Gemma 2-270M weights for GoMLX.
- [x] Implement T5Gemma encoder logic in GoMLX.
- [x] Implement **Mean Pooling** graph: `graph.ReduceMean(encoderHiddenStates, 1)`.
- [x] Verify 1152-dimension output vector.

### Phase 5: MCP Server Implementation
- [x] Choose a Go MCP library (e.g., `github.com/mark3labs/mcp-go`).
- [x] Define the MCP Server with its tools (`search_multimodal`, `ingest_asset`, `get_asset_details`).
- [x] Implement tool handlers that call the underlying RAG logic.
- [x] Set up the standard input/output (stdio) loop for MCP communication.
- [x] Test the server using the MCP Inspector or a compliant client.

### Phase 6: Interface & Verification
- [x] Implement `cmd/mcp-server/main.go` to initialize GoMLX and start the MCP loop.
- [x] Test the full flow: LLM calls `ingest_asset` -> GoMLX embeds -> PG stores.
- [x] Test search: LLM calls `search_multimodal` -> PG finds context -> LLM receives JSON.
- [ ] Benchmark MCP tool-call latency.

### Phase 7: OpenAI-Compatible API Layer
- [x] Implement JWT utility (`pkg/utils/jwt.go`) using EdDSA (Ed25519) from the standard library.
- [x] Define OpenAI-compatible API types (`pkg/api/types.go`).
- [x] Implement JWT middleware and API handlers (`pkg/api/server.go`) using public/private keys.
- [x] Implement `/v1/chat/completions` (RAG-augmented), `/v1/embeddings`, and `/v1/models`.
- [x] Create main entry point `cmd/api/main.go` with Ed25519 key derivation.
- [x] Implement actual Gemma 2 decoder for `/v1/chat/completions`.
- [ ] Add integration tests for OpenAI-compatible endpoints.

### Phase 8: IBM Granite 4.0 350M-H Integration
- [x] Research Granite 4.0 Hybrid (Mamba-2 + Transformer) architecture.
- [x] Implement `pkg/embedder/granite.go` with hybrid layer support.
- [x] Update `pkg/embedder/config.go` with `GraniteHybridConfig`.
- [x] Implement dispatcher in `pkg/embedder/embedder.go`.
- [x] Update API and MCP main entry points to support both models.
- [x] Verify 768-dimension embedding output.
- [x] Performance benchmarking for hybrid inference.

### Phase 9: Database & Migration Utilities
- [x] Implement `db-migrate` utility for re-embedding assets between models.
- [x] Add support for batch metadata modifications during migration.
- [x] Implement paginated database retrieval for large-scale migrations.

### Phase 10: Responses API Implementation
- [x] Define Responses API types in `pkg/api/types.go` (`ResponseRequest`, `ResponseItem`, `ResponseResponse`).
- [x] Implement `HandleResponses` in `pkg/api/server.go` to support the unified interface.
- [x] Map `instructions` and `input` (text/items) to the underlying `DynaGenerate` RAG flow.
- [x] Register the `/v1/responses` route in the API server.
- [ ] Add integration tests for the Responses API endpoint.

## Step-by-Step Details

### DB Creation SQL
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1152)
);
CREATE INDEX ON filesys USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
```

### Normalization Logic
```go
// Normalization Maps [0, 255] to [-1, 1]
images = DivScalar(ConvertDType(rawImages, Float32), 255.0)
normalized = Div(Sub(images, Const(g, 0.5)), Const(g, 0.5))
```

### PG Search Accuracy Tuning
```sql
SET hnsw.ef_search = 100;
```
