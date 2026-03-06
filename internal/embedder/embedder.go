package embedder

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// EmbedMultimodalGraph builds the full GoMLX graph for text+image encoding.
func EmbedMultimodalGraph(ctx *context.Context, textIds, imagePixels *Node, cfg *Config) *Node {
	// 1. Preprocess Images
	normImages := PreprocessImageGraph(ctx, imagePixels)

	// 2. Vision Encoder (SigLIP)
	visualTokens := SigLIPVisionEncoder(ctx, normImages, cfg.Encoder.VisionConfig)

	// 3. Vision-to-Text Projection
	ctx = ctx.In("vision_projection")
	visualTokens = layers.Dense(ctx, visualTokens, true, cfg.Encoder.TextConfig.HiddenSize)

	// 4. Text Embedding
	textCtx := ctx.In("text_embedding")
	textTokens := layers.Embedding(textCtx, textIds, dtypes.Float32, cfg.VocabSize, cfg.Encoder.TextConfig.HiddenSize)

	// 5. Concatenate Tokens
	combinedTokens := Concatenate([]*Node{visualTokens, textTokens}, 1)

	// 6. Gemma 3 Multimodal Encoder
	encoderHiddenStates := Gemma3Encoder(ctx, combinedTokens, cfg.Encoder.TextConfig)

	// 7. Mean Pooling
	return MeanPoolingGraph(ctx, encoderHiddenStates)
}

// GenerateMultimodalGraph builds the full GoMLX graph for text+image conditional generation.
func GenerateMultimodalGraph(ctx *context.Context, textIds, imagePixels, decoderIds *Node, cfg *Config) *Node {
	// 1. Preprocess Images
	normImages := PreprocessImageGraph(ctx, imagePixels)

	// 2. Vision Encoder (SigLIP)
	visualTokens := SigLIPVisionEncoder(ctx, normImages, cfg.Encoder.VisionConfig)

	// 3. Vision-to-Text Projection
	ctx = ctx.In("vision_projection")
	visualTokens = layers.Dense(ctx, visualTokens, true, cfg.Encoder.TextConfig.HiddenSize)

	// 4. Encoder Text Embedding
	textCtx := ctx.In("text_embedding")
	encoderTextTokens := layers.Embedding(textCtx, textIds, dtypes.Float32, cfg.VocabSize, cfg.Encoder.TextConfig.HiddenSize)

	// 5. Concatenate Tokens
	combinedTokens := Concatenate([]*Node{visualTokens, encoderTextTokens}, 1)

	// 6. Gemma 3 Multimodal Encoder
	encoderHiddenStates := Gemma3Encoder(ctx, combinedTokens, cfg.Encoder.TextConfig)

	// 7. Decoder Text Embedding
	decoderTextCtx := ctx.In("decoder_text_embedding")
	decoderTokens := layers.Embedding(decoderTextCtx, decoderIds, dtypes.Float32, cfg.VocabSize, cfg.Decoder.HiddenSize)

	// 8. Gemma 3 Decoder
	decoderHiddenStates := Gemma3Decoder(ctx, decoderTokens, encoderHiddenStates, cfg.Decoder)

	// 9. Output Head (tied to embeddings)
	logits := layers.Dense(textCtx, decoderHiddenStates, false, cfg.VocabSize)
	return logits
}
