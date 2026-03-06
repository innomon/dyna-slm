# DYNA-SLM Implementation Plan: Dynamic Data Tuned Small Language Model

This plan outlines the implementation steps for the Dyna-SLM architecture, transitioning from traditional RAG to deep-integrated retrieval within the model inference loop.

## 1. Implementation Checklist

### Phase 1: Configuration & Infrastructure
- [ ] Define the `DynaModelConfig` struct (including `WeightsPath` and `PreFilterSQL`).
- [ ] Implement a configuration loader for multiple model variants.
- [ ] Refactor `internal/db` to support dynamic SQL generation with pre-filters (metadata/sub-path).
- [ ] Refactor `internal/db` to support multiple tables/databases dynamically based on embedding dimension.
- [ ] Update SQL schema to include dimension-specific tables (e.g., `filesys_640`, `filesys_768`).

### Phase 2: Embedded RAG Layer (GoMLX)
- [ ] Modify `internal/embedder` to load weights from `WeightsPath` specified in the configuration.
- [ ] Modify `internal/embedder` to expose intermediate encoder hidden states.
- [ ] Implement the **Latent Retrieval Step**:
    - [ ] Run the encoder graph.
    - [ ] Extract the mean-pooled vector.
    - [ ] Call the vector DB using the latent vector and the model-specific `PreFilterSQL`.
- [ ] Implement the **Fusion Transformer** graph:
    - [ ] Input: Encoder latent states + Retrieved vectors (as tokens/states).
    - [ ] Output: Combined hidden representation.
- [ ] Integrate the Fusion Transformer output into the decoder input.

### Phase 3: Dynamic Model Registry
- [ ] Implement a `ModelRegistry` that initializes and holds multiple GoMLX models (Gemma 3, Granite) based on the config file.
- [ ] Update `internal/api` to dynamically populate `/v1/models` from the registry.
- [ ] Update `/v1/chat/completions` to route requests to the correct model variant and perform the embedded RAG flow.

### Phase 4: Reference Management
- [ ] Modify the RAG orchestrator to track and return record paths through the entire pipeline.
- [ ] Implement a "Post-Processor" that appends "References" to the final generated text.

### Phase 5: Specialized MCP Servers
- [ ] Create a multi-instance MCP server entry point.
- [ ] Allow passing model configuration (encoder/dim/db) via environment variables or CLI flags.
- [ ] Define Dyna-specific MCP tools: `dyna_search`, `dyna_ingest`, `dyna_list_variants`.

### Phase 6: Validation & Verification
- [ ] Create test scripts for each model variant.
- [ ] Verify that embeddings from the encoder correctly retrieve relevant records from their respective dimension-specific tables.
- [ ] Verify the "References" section in the chat output.
- [ ] Benchmark latency of the embedded RAG layer versus standard RAG.

## 2. Step-by-Step Details

### Configuration Format (`models.json`)
```json
[
  {
    "name": "dyna-gemma3-270m",
    "architecture": "gemma3",
    "weights_path": "./models/t5gemma-2-270m/model.bin",
    "embedding_dimension": 640,
    "database_name": "gemma3_vdb",
    "pre_filter_sql": "path LIKE '/docs/legal/%'",
    "k": 5
  },
  {
    "name": "dyna-granite-350m",
    "architecture": "granite-hybrid",
    "weights_path": "./models/granite-350m/model.safetensors",
    "embedding_dimension": 768,
    "database_name": "granite_vdb",
    "pre_filter_sql": "metadata->>'author' = 'research_team'",
    "k": 3
  }
]
```

### SQL Update for Dynamic Tables and Filters
```sql
-- Procedure to create dimension-specific tables
CREATE OR REPLACE FUNCTION create_vector_table(table_name TEXT, dim INTEGER)
RETURNS VOID AS $$
BEGIN
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I (
        path TEXT PRIMARY KEY,
        metadata JSONB,
        content BYTEA,
        tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        embedding vector(%L)
    )', table_name, dim);
    
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I_idx ON %I USING hnsw (embedding vector_cosine_ops)', table_name, table_name);
END;
$$ LANGUAGE plpgsql;

-- Conceptual dynamic search query with pre-filter
-- SELECT path, metadata FROM %I WHERE %s ORDER BY embedding <=> %L LIMIT %d;
```

### Fusion Transformer Logic (Conceptual)
The fusion step will likely involve a cross-attention layer where:
-   **Query:** Encoder output sequence.
-   **Key/Value:** Retrieved vectors treated as additional context tokens.
-   **Output:** Enhanced latent states for the decoder.
