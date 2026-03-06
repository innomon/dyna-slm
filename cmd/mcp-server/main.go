package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/innomon/gomlx-pgvect-rag/internal/config"
	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	// Register XLA backend
	_ "github.com/gomlx/gomlx/backends/xla"
)

type SearchMultimodalArgs struct {
	Model          string `json:"model" jsonschema:"The Dyna-SLM model variant to use (e.g., dyna-gemma3-270m)."`
	QueryText      string `json:"query_text" jsonschema:"Textual query for similarity search."`
	QueryImagePath string `json:"query_image_path" jsonschema:"Local path to an image file for visual similarity search."`
	Limit          int    `json:"limit" jsonschema:"Maximum number of results to return (default: 5)."`
}

type IngestAssetArgs struct {
	Model string `json:"model" jsonschema:"The Dyna-SLM model variant to use for embedding (e.g., dyna-gemma3-270m)."`
	Path  string `json:"path" jsonschema:"Local path to the file to ingest."`
}

func main() {
	var configPath string
	flag.StringVar(&configPath, "config", os.Getenv("DYNA_CONFIG"), "Path to models.json configuration")
	flag.Parse()

	if configPath == "" {
		configPath = "models.json"
	}

	// 1. Initialize GoMLX Backend
	backend, err := gomlx_utils.InitializeBackend()
	if err != nil {
		log.Fatalf("GoMLX initialization failed: %v", err)
	}

	// 2. Load Dyna-SLM Configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load models config from %s: %v", configPath, err)
	}

	// 3. Initialize Database Connection
	ctx := context.Background()
	pool, err := db.Connect(ctx)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// 4. Initialize Model Registry
	fmt.Fprintf(os.Stderr, "📂 Initializing Model Registry from %s...\n", configPath)
	registry, err := rag.NewRegistry(ctx, cfg, pool, backend)
	if err != nil {
		log.Fatalf("Failed to initialize model registry: %v", err)
	}

	// 5. Create MCP Server
	server := mcp.NewServer(
		&mcp.Implementation{
			Name:    "dyna-slm-mcp",
			Version: "1.0.0",
		},
		&mcp.ServerOptions{},
	)

	// 6. Register Tools using the top-level generic AddTool
	// search_multimodal
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search_multimodal",
		Description: "Search for relevant text or image assets in the multimodal RAG store using a specific Dyna-SLM variant.",
	}, func(ctx context.Context, request *mcp.CallToolRequest, args SearchMultimodalArgs) (*mcp.CallToolResult, any, error) {
		orch, err := registry.GetOrchestrator(args.Model)
		if err != nil {
			return nil, nil, fmt.Errorf("model error: %w", err)
		}

		limit := args.Limit
		if limit <= 0 {
			limit = 5
		}

		assets, err := orch.Search(ctx, args.QueryText, args.QueryImagePath, limit)
		if err != nil {
			return nil, nil, err
		}

		return nil, assets, nil
	})

	// ingest_asset
	mcp.AddTool(server, &mcp.Tool{
		Name:        "ingest_asset",
		Description: "Ingest a new file (image or text) into the multimodal RAG store using a specific Dyna-SLM variant.",
	}, func(ctx context.Context, request *mcp.CallToolRequest, args IngestAssetArgs) (*mcp.CallToolResult, any, error) {
		orch, err := registry.GetOrchestrator(args.Model)
		if err != nil {
			return nil, nil, fmt.Errorf("model error: %w", err)
		}

		err = orch.Ingest(ctx, args.Path, map[string]interface{}{"source": "mcp-tool", "model": args.Model})
		if err != nil {
			return nil, nil, err
		}

		return nil, fmt.Sprintf("Successfully ingested into %s: %s", args.Model, args.Path), nil
	})

	// list_variants
	mcp.AddTool(server, &mcp.Tool{
		Name:        "list_variants",
		Description: "List all available Dyna-SLM model variants.",
	}, func(ctx context.Context, request *mcp.CallToolRequest, args struct{}) (*mcp.CallToolResult, any, error) {
		return nil, registry.ListModels(), nil
	})

	// 8. Start the MCP Server (stdio)
	fmt.Fprintf(os.Stderr, "🚀 Dyna-SLM MCP Server running on %s\n", backend.Name())
	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}
