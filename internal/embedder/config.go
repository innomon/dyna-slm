package embedder

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config represents the T5Gemma 2 / Gemma 3 architecture parameters.
type Config struct {
	ModelType           string        `json:"model_type"`
	IsEncoderDecoder    bool          `json:"is_encoder_decoder"`
	VocabSize           int           `json:"vocab_size"`
	HiddenSize          int           `json:"hidden_size"`
	NumHiddenLayers     int           `json:"num_hidden_layers"`
	NumAttentionHeads   int           `json:"num_attention_heads"`
	NumKeyValueHeads    int           `json:"num_key_value_heads"`
	HeadDim             int           `json:"head_dim"`
	IntermediateSize    int           `json:"intermediate_size"`
	MaxPositionEmbed    int           `json:"max_position_embeddings"`
	RMSNormEps          float64       `json:"rms_norm_eps"`
	SlidingWindow       int           `json:"sliding_window"`
	
	Decoder             *SubConfig    `json:"decoder"`
	Encoder             *EncoderConfig `json:"encoder"`

	// Granite Hybrid specific config
	GraniteHybrid       *GraniteHybridConfig `json:"granite_hybrid"`
}

type GraniteHybridConfig struct {
	VocabSize           int      `json:"vocab_size"`
	HiddenSize          int      `json:"hidden_size"`
	NumHiddenLayers     int      `json:"num_hidden_layers"`
	LayerTypes          []string `json:"layer_types"` // "mamba" or "attention"
	RMSNormEps          float64  `json:"rms_norm_eps"`
	
	// Attention params
	NumAttentionHeads   int      `json:"num_attention_heads"`
	NumKeyValueHeads    int      `json:"num_key_value_heads"`
	IntermediateSize    int      `json:"intermediate_size"`
	RoPETheta           float64  `json:"rope_theta"`

	// Mamba2 params
	MambaStateDim       int      `json:"mamba_state_dim"`
	MambaConvKernel     int      `json:"mamba_conv_kernel"`
	MambaExpand         int      `json:"mamba_expand"`
	MambaHeads          int      `json:"mamba_heads"`
	MambaHeadDim        int      `json:"mamba_head_dim"`
	MambaChunkSize      int      `json:"mamba_chunk_size"`
}

type SubConfig struct {
	ModelType           string  `json:"model_type"`
	VocabSize           int     `json:"vocab_size"`
	HiddenSize          int     `json:"hidden_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	NumKeyValueHeads    int     `json:"num_key_value_heads"`
	HeadDim             int     `json:"head_dim"`
	IntermediateSize    int     `json:"intermediate_size"`
	RMSNormEps          float64 `json:"rms_norm_eps"`
	SlidingWindow       int     `json:"sliding_window"`
}

type EncoderConfig struct {
	ModelType     string         `json:"model_type"`
	VocabSize     int            `json:"vocab_size"`
	TextConfig    *SubConfig     `json:"text_config"`
	VisionConfig  *VisionConfig  `json:"vision_config"`
}

type VisionConfig struct {
	ModelType           string  `json:"model_type"`
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	ImageSize           int     `json:"image_size"`
	PatchSize           int     `json:"patch_size"`
	LayerNormEps        float64 `json:"layer_norm_eps"`
}

// LoadConfig reads and parses a config.json file.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}
