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

// UpsertAsset inserts or updates an asset and its vector embedding.
func UpsertAsset(ctx context.Context, pool *pgxpool.Pool, asset Asset) error {
	metaJSON, err := json.Marshal(asset.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	vec := pgvector.NewVector(asset.Embedding)

	query := `
		INSERT INTO filesys (path, metadata, content, embedding, tmstamp)
		VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
		ON CONFLICT (path) DO UPDATE SET
			metadata = EXCLUDED.metadata,
			content = EXCLUDED.content,
			embedding = EXCLUDED.embedding,
			tmstamp = CURRENT_TIMESTAMP;
	`

	_, err = pool.Exec(ctx, query, asset.Path, metaJSON, asset.Content, vec)
	if err != nil {
		return fmt.Errorf("failed to upsert asset: %w", err)
	}

	return nil
}

// SearchSimilar retrieves the top K most similar assets based on the query vector.
func SearchSimilar(ctx context.Context, pool *pgxpool.Pool, queryVec []float32, limit int) ([]Asset, error) {
	vec := pgvector.NewVector(queryVec)

	// Tuning accuracy at search time
	_, _ = pool.Exec(ctx, "SET hnsw.ef_search = 100")

	query := `
		SELECT path, metadata, content, (embedding <=> $1) as distance
		FROM filesys
		ORDER BY distance
		LIMIT $2;
	`

	rows, err := pool.Query(ctx, query, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query similar assets: %w", err)
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
