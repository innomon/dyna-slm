package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// DynaModelConfig defines the configuration for a Dyna-SLM model variant.
type DynaModelConfig struct {
	Name               string `json:"name"`
	Architecture       string `json:"architecture"`
	WeightsPath        string `json:"weights_path"`
	EmbeddingDimension int    `json:"embedding_dimension"`
	DatabaseName       string `json:"database_name"`
	PreFilterSQL       string `json:"pre_filter_sql"`
	K                  int    `json:"k"`
}

// Config is the top-level configuration structure.
type Config struct {
	Models []DynaModelConfig `json:"models"`
}

// LoadConfig loads the model configuration from a JSON file.
func LoadConfig(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	var cfg Config
	if err := json.NewDecoder(file).Decode(&cfg); err != nil {
		return nil, fmt.Errorf("failed to decode config file: %w", err)
	}

	return &cfg, nil
}
