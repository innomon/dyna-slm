package embedder

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// EmbedMultimodalGraph builds the full GoMLX graph for text+image encoding.
func EmbedMultimodalGraph(ctx *context.Context, textIds, imagePixels *Node) *Node {
	// 1. Preprocess Images
	normImages := PreprocessImageGraph(ctx, imagePixels)

	// 2. Vision Encoder (SigLIP)
	visualTokens := SigLIPVisionEncoder(ctx, normImages)

	// 3. Vision-to-Text Projection
	ctx = ctx.In("vision_projection")
	visualTokens = layers.Dense(ctx, visualTokens, true, 640)

	// 4. Text Embedding
	// Updated signature: (ctx, input, dtype, vocabSize, embeddingDim)
	ctx = ctx.In("text_embedding")
	textTokens := layers.Embedding(ctx, textIds, dtypes.Float32, 262144, 640)

	// 5. Concatenate Tokens
	combinedTokens := Concatenate([]*Node{visualTokens, textTokens}, 1)

	// 6. Gemma 3 Multimodal Encoder
	encoderHiddenStates := Gemma3Encoder(ctx, combinedTokens)

	// 7. Mean Pooling
	return MeanPoolingGraph(ctx, encoderHiddenStates)
}
