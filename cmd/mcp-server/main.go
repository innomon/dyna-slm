package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	// Register XLA backend
	_ "github.com/gomlx/gomlx/backends/xla"
)

type SearchMultimodalArgs struct {
	QueryText      string `json:"query_text" jsonschema:"Textual query for similarity search."`
	QueryImagePath string `json:"query_image_path" jsonschema:"Local path to an image file for visual similarity search."`
	Limit          int    `json:"limit" jsonschema:"Maximum number of results to return (default: 5)."`
}

type IngestAssetArgs struct {
	Path string `json:"path" jsonschema:"Local path to the file to ingest."`
}

func main() {
	var weightsDir string
	flag.StringVar(&weightsDir, "weights", os.Getenv("MODEL_WEIGHTS_DIR"), "Directory containing .safetensors weights")
	flag.Parse()

	// 1. Initialize GoMLX Backend
	backend, err := gomlx_utils.InitializeBackend()
	if err != nil {
		log.Fatalf("GoMLX initialization failed: %v", err)
	}

	// 2. Initialize Model and Load Weights
	model := gomlx_utils.NewModel(backend)
	if weightsDir != "" {
		fmt.Fprintf(os.Stderr, "📂 Loading weights from: %s\n", weightsDir)
		if err := model.LoadSafetensors(weightsDir); err != nil {
			log.Fatalf("Failed to load weights: %v", err)
		}
		// Compile the graph once weights are loaded
		fmt.Fprintf(os.Stderr, "🛠️  Compiling GoMLX graph...\n")
		model.CompileEmbed(embedder.EmbedMultimodalGraph)
	} else {
		fmt.Fprintf(os.Stderr, "⚠️  No weights directory provided. Embeddings will be stubs.\n")
	}

	// 3. Initialize Tokenizer
	var tk *utils.Tokenizer
	if weightsDir != "" {
		tokenizerPath := filepath.Join(weightsDir, "tokenizer.json")
		tk, err = utils.NewTokenizer(tokenizerPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Failed to load tokenizer from %s: %v\n", tokenizerPath, err)
		}
	}

	// 4. Initialize Database Connection
	ctx := context.Background()
	pool, err := db.Connect(ctx)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// 5. Initialize Orchestrator
	orchestrator := &rag.Orchestrator{
		DB:        pool,
		Model:     model,
		Tokenizer: tk,
	}

	// 6. Create MCP Server
	server := mcp.NewServer(
		&mcp.Implementation{
			Name:    "gomlx-pgvect-rag",
			Version: "1.0.0",
		},
		&mcp.ServerOptions{},
	)

	// 7. Register Tools using the top-level generic AddTool
	// search_multimodal
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search_multimodal",
		Description: "Search for relevant text or image assets in the multimodal RAG store.",
	}, func(ctx context.Context, request *mcp.CallToolRequest, args SearchMultimodalArgs) (*mcp.CallToolResult, any, error) {
		limit := args.Limit
		if limit <= 0 {
			limit = 5
		}

		assets, err := orchestrator.Search(ctx, args.QueryText, args.QueryImagePath, limit)
		if err != nil {
			return nil, nil, err
		}

		return nil, assets, nil
	})

	// ingest_asset
	mcp.AddTool(server, &mcp.Tool{
		Name:        "ingest_asset",
		Description: "Ingest a new file (image or text) into the multimodal RAG store.",
	}, func(ctx context.Context, request *mcp.CallToolRequest, args IngestAssetArgs) (*mcp.CallToolResult, any, error) {
		err := orchestrator.Ingest(ctx, args.Path, map[string]interface{}{"source": "mcp-tool"})
		if err != nil {
			return nil, nil, err
		}

		return nil, fmt.Sprintf("Successfully ingested: %s", args.Path), nil
	})

	// 8. Start the MCP Server (stdio)
	fmt.Fprintf(os.Stderr, "🚀 gomlx-pgvect-rag MCP Server running on %s\n", backend.Name())
	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}
