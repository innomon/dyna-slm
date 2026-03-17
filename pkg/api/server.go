package api

import (
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/innomon/gomlx-pgvect-rag/pkg/rag"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
)

type Server struct {
	Registry *rag.Registry
	PrivKey  ed25519.PrivateKey
	PubKey   ed25519.PublicKey
}

func NewServer(registry *rag.Registry, privKey ed25519.PrivateKey) *Server {
	return &Server{
		Registry: registry,
		PrivKey:  privKey,
		PubKey:   privKey.Public().(ed25519.PublicKey),
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

	// Select the correct orchestrator from the registry
	orch, err := s.Registry.GetOrchestrator(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("Model not found: %v", err), http.StatusNotFound)
		return
	}

	lastMessage := ""
	if len(req.Messages) > 0 {
		lastMessage = req.Messages[len(req.Messages)-1].Content
	}

	// Call the actual dyna generator (Embedded RAG)
	responseText, err := orch.DynaGenerate(r.Context(), lastMessage, "", 128)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generation failed: %v", err), http.StatusInternalServerError)
		return
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
					Content: responseText,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Handler for /v1/responses (New)
func (s *Server) HandleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	// Select the correct orchestrator from the registry
	orch, err := s.Registry.GetOrchestrator(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("Model not found: %v", err), http.StatusNotFound)
		return
	}

	// Extract text from input
	inputText := ""
	switch v := req.Input.(type) {
	case string:
		inputText = v
	case []interface{}:
		// Parse ResponseItem array from generic map
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if content, ok := m["content"].([]interface{}); ok {
					for _, c := range content {
						if cp, ok := c.(map[string]interface{}); ok {
							if cp["type"] == "text" {
								if text, ok := cp["text"].(string); ok {
									inputText += text + "\n"
								}
							}
						}
					}
				}
			}
		}
	}

	// Prepend instructions if provided
	fullPrompt := ""
	if req.Instructions != "" {
		fullPrompt = req.Instructions + "\n\n"
	}
	fullPrompt += inputText

	// Call the actual dyna generator (Embedded RAG)
	responseText, err := orch.DynaGenerate(r.Context(), fullPrompt, "", 128)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	resp := ResponseResponse{
		Id:      "resp-" + time.Now().Format("20060102150405"),
		Object:  "response",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Output: []ResponseItem{
			{
				Id:   "item-" + time.Now().Format("20060102150405-01"),
				Type: "message",
				Role: "assistant",
				Content: []ResponseContent{
					{
						Type: "text",
						Text: responseText,
					},
				},
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

	// Select the correct orchestrator from the registry
	orch, err := s.Registry.GetOrchestrator(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("Model not found: %v", err), http.StatusNotFound)
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
		tokens, err := orch.Tokenizer.Encode(input, true)
		if err != nil {
			http.Error(w, "Tokenization failed", http.StatusInternalServerError)
			return
		}
		
		// Create a zero image tensor as T5Gemma 2 is multimodal but we're doing text embedding
		imgT := orch.OrchestratorZeroImageTensor()

		vec, err := orch.Model.Embed(tokens, imgT)
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
	modelNames := s.Registry.ListModels()
	modelInfos := make([]ModelInfo, 0, len(modelNames))
	
	for _, name := range modelNames {
		modelInfos = append(modelInfos, ModelInfo{
			Id:      name,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "dyna-slm",
		})
	}

	resp := ModelList{
		Object: "list",
		Data:   modelInfos,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// RegisterHandlers sets up the routes
func (s *Server) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/v1/chat/completions", s.AuthMiddleware(s.HandleChatCompletions))
	mux.HandleFunc("/v1/responses", s.AuthMiddleware(s.HandleResponses))
	mux.HandleFunc("/v1/embeddings", s.AuthMiddleware(s.HandleEmbeddings))
	mux.HandleFunc("/v1/models", s.AuthMiddleware(s.HandleModels))
	
	// Open endpoint to get a token for demo/testing
	mux.HandleFunc("/auth/token", func(w http.ResponseWriter, r *http.Request) {
		token, _ := utils.GenerateJWT(s.PrivKey, "user-123", 24*time.Hour)
		fmt.Fprintf(w, "%s", token)
	})
}
