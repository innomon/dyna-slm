package embedder

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// EmbedGraph builds the GoMLX graph for encoding, dispatching to the correct model type.
func EmbedGraph(ctx *context.Context, textIds, imagePixels *Node, cfg *Config) *Node {
	if cfg.ModelType == "granite-hybrid" {
		return GraniteHybridEmbedderGraph(ctx, textIds, cfg)
	}
	return EmbedMultimodalGraph(ctx, textIds, imagePixels, cfg)
}

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

// DynaEncoderGraph returns (pooled_embedding, encoder_hidden_states).
func DynaEncoderGraph(ctx *context.Context, textIds, imagePixels *Node, cfg *Config) []*Node {
	// 1-5. Preprocess and Concatenate (Same as EmbedMultimodalGraph)
	normImages := PreprocessImageGraph(ctx, imagePixels)
	visualTokens := SigLIPVisionEncoder(ctx, normImages, cfg.Encoder.VisionConfig)
	
	ctxVisionProj := ctx.In("vision_projection")
	visualTokens = layers.Dense(ctxVisionProj, visualTokens, true, cfg.Encoder.TextConfig.HiddenSize)
	
	textCtx := ctx.In("text_embedding")
	textTokens := layers.Embedding(textCtx, textIds, dtypes.Float32, cfg.VocabSize, cfg.Encoder.TextConfig.HiddenSize)
	
	combinedTokens := Concatenate([]*Node{visualTokens, textTokens}, 1)

	// 6. Gemma 3 Multimodal Encoder
	encoderHiddenStates := Gemma3Encoder(ctx, combinedTokens, cfg.Encoder.TextConfig)

	// 7. Mean Pooling for search vector
	pooled := MeanPoolingGraph(ctx, encoderHiddenStates)
	
	return []*Node{pooled, encoderHiddenStates}
}

// DynaFusionDecoderGraph builds the graph for Fusion + Decoding.
func DynaFusionDecoderGraph(ctx *context.Context, encoderHiddenStates, retrievedVectors, decoderIds *Node, cfg *Config) *Node {
	// 1. Fusion Transformer (Encoder States + Retrieved Vectors)
	fusedStates := FusionTransformer(ctx, encoderHiddenStates, retrievedVectors, cfg.Encoder.TextConfig)
	
	// 2. Decoder Text Embedding
	decoderTextCtx := ctx.In("decoder_text_embedding")
	decoderTokens := layers.Embedding(decoderTextCtx, decoderIds, dtypes.Float32, cfg.VocabSize, cfg.Decoder.HiddenSize)

	// 3. Gemma 3 Decoder
	decoderHiddenStates := Gemma3Decoder(ctx, decoderTokens, fusedStates, cfg.Decoder)

	// 4. Output Head
	textCtx := ctx.In("text_embedding") // Tie weights if needed
	logits := layers.Dense(textCtx, decoderHiddenStates, false, cfg.VocabSize)
	return logits
}

// GenerateGraph builds the GoMLX graph for generation, dispatching based on model type.
func GenerateGraph(ctx *context.Context, textIds, imagePixels, decoderIds *Node, cfg *Config) *Node {
	if cfg.ModelType == "granite-hybrid" {
		return GraniteHybridGenerateGraph(ctx, textIds, decoderIds, cfg)
	}
	return GenerateMultimodalGraph(ctx, textIds, imagePixels, decoderIds, cfg)
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
