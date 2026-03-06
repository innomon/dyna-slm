package rag

import (
	"context"
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Orchestrator manages the RAG pipeline.
type Orchestrator struct {
	DB        *pgxpool.Pool
	Model     *gomlx_utils.Model
	Tokenizer *utils.Tokenizer
}

// OrchestratorZeroImageTensor returns a zero-filled image tensor for text-only operations.
func (o *Orchestrator) OrchestratorZeroImageTensor() *tensors.Tensor {
	return tensors.FromFlatDataAndDimensions(make([]float32, 896*896*3), 1, 896, 896, 3)
}

// Search retrieves relevant assets based on text or image input.
func (o *Orchestrator) Search(ctx context.Context, text string, imagePath string, limit int) ([]db.Asset, error) {
	// 1. Prepare Inputs
	var imgT *tensors.Tensor
	var tokens []uint32
	var err error

	if imagePath != "" {
		imgT, err = embedder.LoadImageAsTensor(imagePath)
		if err != nil {
			return nil, err
		}
	} else {
		imgT = o.OrchestratorZeroImageTensor()
	}

	if text != "" {
		tokens, err = o.Tokenizer.Encode(text, true)
		if err != nil {
			return nil, err
		}
	} else {
		tokens = []uint32{0} // Single pad/empty token
	}

	// 2. Generate Embedding via GoMLX
	queryVec, err := o.Model.Embed(tokens, imgT)
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed: %w", err)
	}

	// 3. Query Vector DB
	results, err := db.SearchSimilar(ctx, o.DB, queryVec, limit)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	return results, nil
}

// Ingest adds a new asset to the RAG store.
func (o *Orchestrator) Ingest(ctx context.Context, path string, metadata map[string]interface{}) error {
	// 1. Generate embedding
	imgT, err := embedder.LoadImageAsTensor(path)
	if err != nil {
		return fmt.Errorf("failed to load file for ingestion: %w", err)
	}
	
	tokens := []uint32{0}
	vec, err := o.Model.Embed(tokens, imgT)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for ingestion: %w", err)
	}

	// 2. Upsert into DB
	asset := db.Asset{
		Path:      path,
		Metadata:  metadata,
		Embedding: vec,
	}
	
	return db.UpsertAsset(ctx, o.DB, asset)
}

// Generate uses the T5Gemma 2 decoder to produce a response, augmented by RAG context.
func (o *Orchestrator) Generate(ctx context.Context, text string, imagePath string, maxTokens int) (string, error) {
	// 1. RAG: Search for context
	results, err := o.Search(ctx, text, imagePath, 3)
	if err != nil {
		return "", fmt.Errorf("search failed during generation: %w", err)
	}

	// 2. Build Prompt with context
	prompt := "Context:\n"
	for _, res := range results {
		prompt += fmt.Sprintf("- Found in %s: %v\n", res.Path, res.Metadata)
	}
	prompt += "\nUser: " + text + "\nAssistant: "

	// 3. Prepare Inputs
	var imgT *tensors.Tensor
	if imagePath != "" {
		imgT, err = embedder.LoadImageAsTensor(imagePath)
		if err != nil {
			return "", err
		}
	} else {
		imgT = o.OrchestratorZeroImageTensor()
	}

	textIds, err := o.Tokenizer.Encode(prompt, true)
	if err != nil {
		return "", err
	}

	// 4. Generation Loop (Greedy)
	decoderIds := []uint32{2} // BOS token
	
	for i := 0; i < maxTokens; i++ {
		nextId, err := o.Model.GenerateStep(textIds, imgT, decoderIds)
		if err != nil {
			return "", err
		}

		if nextId == 1 { // EOS token
			break
		}

		decoderIds = append(decoderIds, nextId)
	}

	// 5. Decode response
	response, err := o.Tokenizer.Decode(decoderIds[1:]) // Skip BOS
	if err != nil {
		return "", err
	}

	return response, nil
}
