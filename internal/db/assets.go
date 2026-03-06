package db

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// Asset represents a file and its multimodal metadata.
type Asset struct {
	Path      string
	Metadata  map[string]interface{}
	Content   []byte
	Embedding []float32
}

// UpsertAsset inserts or updates an asset and its vector embedding into a specific table.
func UpsertAsset(ctx context.Context, pool *pgxpool.Pool, tableName string, asset Asset) error {
	metaJSON, err := json.Marshal(asset.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	vec := pgvector.NewVector(asset.Embedding)

	// Use fmt.Fprintf to build the query safely with dynamic table name (identifiers)
	query := fmt.Sprintf(`
		INSERT INTO %s (path, metadata, content, embedding, tmstamp)
		VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
		ON CONFLICT (path) DO UPDATE SET
			metadata = EXCLUDED.metadata,
			content = EXCLUDED.content,
			embedding = EXCLUDED.embedding,
			tmstamp = CURRENT_TIMESTAMP;
	`, tableName)

	_, err = pool.Exec(ctx, query, asset.Path, metaJSON, asset.Content, vec)
	if err != nil {
		return fmt.Errorf("failed to upsert asset into %s: %w", tableName, err)
	}

	return nil
}

// SearchSimilar retrieves the top K most similar assets based on the query vector, 
// using a specific table and an optional pre-filter SQL.
func SearchSimilar(ctx context.Context, pool *pgxpool.Pool, tableName string, preFilterSQL string, queryVec []float32, limit int) ([]Asset, error) {
	vec := pgvector.NewVector(queryVec)

	// Tuning accuracy at search time
	_, _ = pool.Exec(ctx, "SET hnsw.ef_search = 100")

	// Construct query with optional pre-filter
	whereClause := ""
	if preFilterSQL != "" {
		whereClause = "WHERE " + preFilterSQL
	}

	query := fmt.Sprintf(`
		SELECT path, metadata, content, (embedding <=> $1) as distance
		FROM %s
		%s
		ORDER BY distance
		LIMIT $2;
	`, tableName, whereClause)

	rows, err := pool.Query(ctx, query, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query similar assets from %s: %w", tableName, err)
	}
	defer rows.Close()

	var assets []Asset
	for rows.Next() {
		var a Asset
		var metaRaw []byte
		var distance float64
		if err := rows.Scan(&a.Path, &metaRaw, &a.Content, &distance); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}
		if err := json.Unmarshal(metaRaw, &a.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
		assets = append(assets, a)
	}

	return assets, nil
}

// InitializeTable ensures a dimension-specific table exists with the correct HNSW index.
func InitializeTable(ctx context.Context, pool *pgxpool.Pool, tableName string, dim int) error {
	query := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			path TEXT PRIMARY KEY,
			metadata JSONB,
			content BYTEA,
			tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
			embedding vector(%d)
		);
	`, tableName, dim)

	if _, err := pool.Exec(ctx, query); err != nil {
		return fmt.Errorf("failed to create table %s: %w", tableName, err)
	}

	indexName := fmt.Sprintf("idx_%s_embedding", tableName)
	indexQuery := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s ON %s 
		USING hnsw (embedding vector_cosine_ops)
		WITH (m = 16, ef_construction = 128);
	`, indexName, tableName)

	if _, err := pool.Exec(ctx, indexQuery); err != nil {
		return fmt.Errorf("failed to create index on %s: %w", tableName, err)
	}

	return nil
}
