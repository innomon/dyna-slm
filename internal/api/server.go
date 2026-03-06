package api

import (
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/innomon/gomlx-pgvect-rag/internal/rag"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
)

type Server struct {
	Orchestrator *rag.Orchestrator
	PrivKey      ed25519.PrivateKey
	PubKey       ed25519.PublicKey
}

func NewServer(orch *rag.Orchestrator, privKey ed25519.PrivateKey) *Server {
	return &Server{
		Orchestrator: orch,
		PrivKey:      privKey,
		PubKey:       privKey.Public().(ed25519.PublicKey),
	}
}

// Middleware for JWT authentication
func (s *Server) AuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Unauthorized: Missing Authorization header", http.StatusUnauthorized)
			return
		}

		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
			http.Error(w, "Unauthorized: Invalid Authorization format", http.StatusUnauthorized)
			return
		}

		_, err := utils.ValidateJWT(s.PubKey, parts[1])
		if err != nil {
			http.Error(w, fmt.Sprintf("Unauthorized: %v", err), http.StatusUnauthorized)
			return
		}

		next(w, r)
	}
}

// Handler for /v1/chat/completions
func (s *Server) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	// For now, return a simple RAG-augmented response if search finds something, 
	// otherwise a static response since the decoder is not yet implemented.
	// In a real scenario, this would call Orchestrator.Generate(...)
	
	lastMessage := ""
	if len(req.Messages) > 0 {
		lastMessage = req.Messages[len(req.Messages)-1].Content
	}

	// Perform search to show RAG capability
	results, _ := s.Orchestrator.Search(r.Context(), lastMessage, "", 3)
	
	contextStr := ""
	if len(results) > 0 {
		contextStr = "\n\nContext found:\n"
		for _, res := range results {
			contextStr += fmt.Sprintf("- %s\n", res.Path)
		}
	}

	resp := ChatCompletionResponse{
		Id:      "chatcmpl-" + time.Now().Format("20060102150405"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello! I am the Dyna-SLM RAG assistant. " + contextStr,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Handler for /v1/embeddings
func (s *Server) HandleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputs = append(inputs, s)
			}
		}
	default:
		http.Error(w, "Invalid input format", http.StatusBadRequest)
		return
	}

	resp := EmbeddingResponse{
		Object: "list",
		Model:  req.Model,
		Data:   make([]EmbeddingData, 0, len(inputs)),
	}

	for i, input := range inputs {
		// Orchestrator uses its internal GoMLX model to generate embeddings
		// Search uses Embed internally, but we can expose a direct Embed method in Orchestrator if needed.
		// For now, let's use the Model.Embed directly through a helper if possible or add it to Orchestrator.
		
		tokens, err := s.Orchestrator.Tokenizer.Encode(input, true)
		if err != nil {
			http.Error(w, "Tokenization failed", http.StatusInternalServerError)
			return
		}
		
		// Create a zero image tensor as T5Gemma 2 is multimodal but we're doing text embedding
		imgT := s.Orchestrator.OrchestratorZeroImageTensor()

		vec, err := s.Orchestrator.Model.Embed(tokens, imgT)
		if err != nil {
			http.Error(w, "Embedding failed", http.StatusInternalServerError)
			return
		}

		resp.Data = append(resp.Data, EmbeddingData{
			Object:    "embedding",
			Embedding: vec,
			Index:     i,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Handler for /v1/models
func (s *Server) HandleModels(w http.ResponseWriter, r *http.Request) {
	resp := ModelList{
		Object: "list",
		Data: []ModelInfo{
			{
				Id:      "t5gemma-2-270m",
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "gomlx-pgvect-rag",
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// RegisterHandlers sets up the routes
func (s *Server) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/v1/chat/completions", s.AuthMiddleware(s.HandleChatCompletions))
	mux.HandleFunc("/v1/embeddings", s.AuthMiddleware(s.HandleEmbeddings))
	mux.HandleFunc("/v1/models", s.AuthMiddleware(s.HandleModels))
	
	// Open endpoint to get a token for demo/testing
	mux.HandleFunc("/auth/token", func(w http.ResponseWriter, r *http.Request) {
		token, _ := utils.GenerateJWT(s.PrivKey, "user-123", 24*time.Hour)
		fmt.Fprintf(w, "%s", token)
	})
}
