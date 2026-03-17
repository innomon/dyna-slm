package rag

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/gomlx/gomlx/backends"
	"github.com/innomon/gomlx-pgvect-rag/pkg/config"
	"github.com/innomon/gomlx-pgvect-rag/pkg/db"
	"github.com/innomon/gomlx-pgvect-rag/pkg/embedder"
	"github.com/innomon/gomlx-pgvect-rag/pkg/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Registry manages multiple Dyna-SLM model variants.
type Registry struct {
	Orchestrators map[string]*Orchestrator
}

// NewRegistry initializes a Registry from the given configuration.
func NewRegistry(ctx context.Context, cfg *config.Config, pool *pgxpool.Pool, backend backends.Backend) (*Registry, error) {
	reg := &Registry{
		Orchestrators: make(map[string]*Orchestrator),
	}

	for _, mCfg := range cfg.Models {
		// 1. Initialize DB Table for this model
		if err := db.InitializeTable(ctx, pool, mCfg.DatabaseName, mCfg.EmbeddingDimension); err != nil {
			return nil, fmt.Errorf("failed to initialize table %s: %w", mCfg.DatabaseName, err)
		}

		// 2. Load Model Config (for GoMLX)
		// We need to know which architecture-specific config to load.
		// For now, let's assume architecture-specific config is alongside weights.
		modelCfgPath := filepath.Join(filepath.Dir(mCfg.WeightsPath), "config.json")
		embedderCfg, err := embedder.LoadConfig(modelCfgPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load embedder config from %s: %w", modelCfgPath, err)
		}

		// 3. Initialize Model
		model := gomlx_utils.NewModel(backend, embedderCfg)
		if err := model.LoadSafetensors(filepath.Dir(mCfg.WeightsPath)); err != nil {
			return nil, fmt.Errorf("failed to load weights for %s: %w", mCfg.Name, err)
		}

		// 4. Compile Graphs
		model.CompileEmbed(embedder.EmbedGraph)
		model.CompileGenerate(embedder.GenerateGraph)
		model.CompileDynaEncoder(embedder.DynaEncoderGraph)
		model.CompileDynaFusionDecoder(embedder.DynaFusionDecoderGraph)

		// 5. Load Tokenizer
		tokenizerPath := filepath.Join(filepath.Dir(mCfg.WeightsPath), "tokenizer.json")
		tk, err := utils.NewTokenizer(tokenizerPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load tokenizer for %s: %w", mCfg.Name, err)
		}

		// 6. Create Orchestrator
		orch := &Orchestrator{
			DB:        pool,
			Model:     model,
			Tokenizer: tk,
			Config:    mCfg,
		}

		reg.Orchestrators[mCfg.Name] = orch
	}

	return reg, nil
}

// GetOrchestrator returns the orchestrator for the given model name.
func (r *Registry) GetOrchestrator(name string) (*Orchestrator, error) {
	orch, ok := r.Orchestrators[name]
	if !ok {
		return nil, fmt.Errorf("model %s not found in registry", name)
	}
	return orch, nil
}

// ListModels returns a list of available model names.
func (r *Registry) ListModels() []string {
	models := make([]string, 0, len(r.Orchestrators))
	for name := range r.Orchestrators {
		models = append(models, name)
	}
	return models
}
