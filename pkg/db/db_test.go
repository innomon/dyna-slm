package db

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestDatabaseIntegration(t *testing.T) {
	// Skip if DATABASE_URL is not set
	connStr := os.Getenv("DATABASE_URL")
	if connStr == "" {
		t.Skip("Skipping integration test: DATABASE_URL not set")
	}

	ctx := context.Background()
	pool, err := Connect(ctx)
	if err != nil {
		t.Fatalf("Failed to connect to database: %v", err)
	}
	defer pool.Close()

	// 1. Create the schema (in case it doesn't exist)
	setupSQL, err := os.ReadFile("schema.sql")
	if err == nil {
		_, err = pool.Exec(ctx, string(setupSQL))
		if err != nil {
			t.Logf("Warning: Failed to run schema.sql: %v (it might already exist)", err)
		}
	}

	// 2. Test Upsert
	tableName := "filesys"
	mockAsset := Asset{
		Path:     "test/image.jpg",
		Metadata: map[string]interface{}{"type": "test", "label": "cat"},
		Content:  []byte("fake-image-data"),
		// 640 dimensions for T5Gemma 2-270M
		Embedding: make([]float32, 640),
	}
	// Set a few values to make it non-zero
	mockAsset.Embedding[0] = 0.5
	mockAsset.Embedding[1] = -0.5

	err = UpsertAsset(ctx, pool, tableName, mockAsset)
	if err != nil {
		t.Fatalf("UpsertAsset failed: %v", err)
	}
	defer CleanUp(pool, tableName)

	// 3. Test Search
	results, err := SearchSimilar(ctx, pool, tableName, "", mockAsset.Embedding, 1)
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("SearchSimilar returned no results")
	}

	if results[0].Path != mockAsset.Path {
		t.Errorf("Expected path %s, got %s", mockAsset.Path, results[0].Path)
	}
}

func CleanUp(pool *pgxpool.Pool, tableName string) {
	ctx := context.Background()
	query := "DELETE FROM " + tableName + " WHERE path = 'test/image.jpg'"
	_, _ = pool.Exec(ctx, query)
}
