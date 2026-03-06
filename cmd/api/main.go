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

	"github.com/innomon/gomlx-pgvect-rag/internal/api"
	"github.com/innomon/gomlx-pgvect-rag/internal/config"
	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"

	// Register XLA backend
	_ "github.com/gomlx/gomlx/backends/xla"
)

func main() {
	var configPath string
	var port int
	var jwtSecret string

	flag.StringVar(&configPath, "config", os.Getenv("DYNA_CONFIG"), "Path to models.json configuration")
	flag.IntVar(&port, "port", 8080, "Port to listen on")
	flag.StringVar(&jwtSecret, "jwt-secret", os.Getenv("JWT_SECRET"), "Seed for Ed25519 key generation (default: 'dev-secret')")
	flag.Parse()

	if jwtSecret == "" {
		jwtSecret = "dev-secret"
	}

	if configPath == "" {
		configPath = "models.json"
	}

	// 0. Initialize Ed25519 Keys
	seed := sha256.Sum256([]byte(jwtSecret))
	privKey := ed25519.NewKeyFromSeed(seed[:])

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

	// 5. Initialize API Server
	apiServer := api.NewServer(registry, privKey)
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
