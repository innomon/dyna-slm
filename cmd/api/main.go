package main

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/innomon/gomlx-pgvect-rag/internal/api"
	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"

	// Register XLA backend
	_ "github.com/gomlx/gomlx/backends/xla"
)

func main() {
	var weightsDir string
	var port int
	var jwtSecret string

	flag.StringVar(&weightsDir, "weights", os.Getenv("MODEL_WEIGHTS_DIR"), "Directory containing .safetensors weights")
	flag.IntVar(&port, "port", 8080, "Port to listen on")
	flag.StringVar(&jwtSecret, "jwt-secret", os.Getenv("JWT_SECRET"), "Seed for Ed25519 key generation (default: 'dev-secret')")
	flag.Parse()

	if jwtSecret == "" {
		jwtSecret = "dev-secret"
	}

	// 0. Initialize Ed25519 Keys
	seed := sha256.Sum256([]byte(jwtSecret))
	privKey := ed25519.NewKeyFromSeed(seed[:])

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
		fmt.Fprintf(os.Stderr, "🛠️  Compiling GoMLX graphs...\n")
		model.CompileEmbed(embedder.EmbedMultimodalGraph)
		model.CompileGenerate(embedder.GenerateMultimodalGraph)
	} else {
		log.Fatalf("Model weights directory is required. Use -weights or MODEL_WEIGHTS_DIR.")
	}

	// 3. Initialize Tokenizer
	tokenizerPath := filepath.Join(weightsDir, "tokenizer.json")
	tk, err := utils.NewTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer from %s: %v", tokenizerPath, err)
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

	// 6. Initialize API Server
	apiServer := api.NewServer(orchestrator, privKey)
	mux := http.NewServeMux()
	apiServer.RegisterHandlers(mux)

	// 7. Start the HTTP Server
	addr := fmt.Sprintf(":%d", port)
	fmt.Fprintf(os.Stderr, "🚀 Dyna-SLM OpenAI-Compatible API running on %s (GoMLX Backend: %s)\n", addr, backend.Name())
	fmt.Fprintf(os.Stderr, "🔑 JWT Signature: EdDSA (Ed25519 derived from seed)\n")
	
	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("HTTP server failed: %v", err)
	}
}
