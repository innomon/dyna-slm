package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/innomon/gomlx-pgvect-rag/internal/config"
	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func main() {
	var configPath string
	var sourceName string
	var targetName string
	var outputTable string
	var addMeta string
	var removeMeta string

	flag.StringVar(&configPath, "config", "models.json", "Path to models configuration")
	flag.StringVar(&sourceName, "source", "", "Source model name")
	flag.StringVar(&targetName, "target", "", "Target model name")
	flag.StringVar(&outputTable, "output-table", "", "Target database table name")
	flag.StringVar(&addMeta, "add-meta", "", "Metadata to add (key:value, comma separated)")
	flag.StringVar(&removeMeta, "remove-meta", "", "Metadata keys to remove (comma separated)")
	flag.Parse()

	if sourceName == "" || targetName == "" {
		log.Fatal("Source and target model names are required")
	}

	// 1. Initialize GoMLX Backend
	backend, err := gomlx_utils.InitializeBackend()
	if err != nil {
		log.Fatalf("GoMLX initialization failed: %v", err)
	}

	// 2. Load Configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load models config: %v", err)
	}

	// 3. Connect to DB
	ctx := context.Background()
	pool, err := db.Connect(ctx)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// 4. Initialize Registry to load both models
	registry, err := rag.NewRegistry(ctx, cfg, pool, backend)
	if err != nil {
		log.Fatalf("Failed to initialize model registry: %v", err)
	}

	sourceOrch, err := registry.GetOrchestrator(sourceName)
	if err != nil {
		log.Fatalf("Source model %s not found: %v", sourceName, err)
	}

	targetOrch, err := registry.GetOrchestrator(targetName)
	if err != nil {
		log.Fatalf("Target model %s not found: %v", targetName, err)
	}

	if outputTable == "" {
		outputTable = targetOrch.Config.DatabaseName
	}

	// Ensure output table exists
	err = db.InitializeTable(ctx, pool, outputTable, targetOrch.Config.EmbeddingDimension)
	if err != nil {
		log.Fatalf("Failed to initialize output table %s: %v", outputTable, err)
	}

	// Parse metadata changes
	toAdd := make(map[string]string)
	if addMeta != "" {
		parts := strings.Split(addMeta, ",")
		for _, p := range parts {
			kv := strings.SplitN(p, ":", 2)
			if len(kv) == 2 {
				toAdd[kv[0]] = kv[1]
			}
		}
	}
	toRemove := strings.Split(removeMeta, ",")
	if removeMeta == "" {
		toRemove = nil
	}

	// 5. Migrate records
	sourceTable := sourceOrch.Config.DatabaseName
	total, err := db.GetAssetCount(ctx, pool, sourceTable)
	if err != nil {
		log.Fatalf("Failed to get asset count from %s: %v", sourceTable, err)
	}

	fmt.Printf("🔄 Starting migration of %d records from %s (%s) to %s (%s)...\n", 
		total, sourceName, sourceTable, targetName, outputTable)

	batchSize := 10
	for offset := 0; offset < total; offset += batchSize {
		assets, err := db.GetAllAssets(ctx, pool, sourceTable, batchSize, offset)
		if err != nil {
			log.Fatalf("Failed to fetch batch at offset %d: %v", offset, err)
		}

		for _, asset := range assets {
			// A. Re-generate embedding for target model
			var imgT *tensors.Tensor
			var tokens []uint32

			if embedder.IsImage(asset.Content) {
				imgT, err = embedder.LoadImageFromBytes(asset.Content)
				if err != nil {
					log.Printf("⚠️ Warning: Failed to load image for %s: %v, skipping", asset.Path, err)
					continue
				}
				tokens = []uint32{0}
			} else {
				imgT = targetOrch.OrchestratorZeroImageTensor()
				// Assume content is text if not image
				tokens, err = targetOrch.Tokenizer.Encode(string(asset.Content), true)
				if err != nil {
					log.Printf("⚠️ Warning: Failed to tokenize text for %s: %v, skipping", asset.Path, err)
					continue
				}
			}

			newVec, err := targetOrch.Model.Embed(tokens, imgT)
			if err != nil {
				log.Printf("⚠️ Warning: Embedding failed for %s: %v, skipping", asset.Path, err)
				continue
			}

			// B. Modify Metadata
			for k, v := range toAdd {
				asset.Metadata[k] = v
			}
			for _, k := range toRemove {
				delete(asset.Metadata, strings.TrimSpace(k))
			}

			// C. Save to target table
			asset.Embedding = newVec
			err = db.UpsertAsset(ctx, pool, outputTable, asset)
			if err != nil {
				log.Printf("❌ Failed to save %s: %v", asset.Path, err)
			} else {
				fmt.Printf("✅ Migrated: %s\n", asset.Path)
			}
		}
	}

	fmt.Println("✨ Migration complete.")
}
