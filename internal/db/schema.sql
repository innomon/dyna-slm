-- schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1152) -- For T5Gemma 2-270M
);

-- Index metadata
CREATE INDEX IF NOT EXISTS idx_filesys_metadata ON filesys USING GIN (metadata);

-- Index embeddings with HNSW for Cosine Similarity
CREATE INDEX IF NOT EXISTS idx_filesys_embedding ON filesys 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);
