package embedder

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// PreprocessImageGraph takes a raw image tensor [batch, height, width, 3]
// and returns a normalized version ready for the SigLIP encoder.
func PreprocessImageGraph(ctx *context.Context, rawImages *Node) *Node {
	// 1. Ensure float32 precision
	images := ConvertDType(rawImages, dtypes.Float32)

	// 2. Rescale from [0, 255] to [0, 1]
	images = DivScalar(images, 255.0)

	// 3. SigLIP Normalization: (x - 0.5) / 0.5
	mean := Const(rawImages.Graph(), 0.5)
	std := Const(rawImages.Graph(), 0.5)

	normalized := Div(Sub(images, mean), std)

	return normalized
}

// MeanPoolingGraph averages across the sequence dimension (axis 1)
func MeanPoolingGraph(ctx *context.Context, encoderHiddenStates *Node) *Node {
	return ReduceMean(encoderHiddenStates, 1)
}

// SigLIPTransformerBlock is a standard ViT transformer layer.
func SigLIPTransformerBlock(ctx *context.Context, x *Node, numHeads, intermediateDim int) *Node {
	// 1. Pre-Attention LayerNorm
	normX := layers.LayerNormalization(ctx.In("pre_attention_norm"), x, -1).Done()

	// 2. Attention with Residual
	attn := layers.MultiHeadAttention(ctx.In("attention"), normX, normX, normX, numHeads, -1).Done()
	x = Add(x, attn)

	// 3. Pre-MLP LayerNorm
	normX = layers.LayerNormalization(ctx.In("pre_mlp_norm"), x, -1).Done()

	// 4. MLP with Residual
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	mlpOut := layers.Dense(ctx.In("mlp/gate"), normX, true, intermediateDim)
	mlpOut = activations.Gelu(mlpOut)
	mlpOut = layers.Dense(ctx.In("mlp/down"), mlpOut, true, hiddenDim)
	x = Add(x, mlpOut)

	return x
}

// SigLIPVisionEncoder processes image tokens based on configuration.
func SigLIPVisionEncoder(ctx *context.Context, x *Node, cfg *VisionConfig) *Node {
	ctx = ctx.In("siglip_vision")
	g := x.Graph()
	
	// 1. Patch Embedding (Convolution)
	x = layers.Convolution(ctx.In("patch_embed"), x).
		Filters(cfg.HiddenSize).
		KernelSize(cfg.PatchSize).
		Strides(cfg.PatchSize).
		Done()
		
	// 2. Flatten and add Positional Embeddings
	batchSize := x.Shape().Dimensions[0]
	numPatches := (cfg.ImageSize / cfg.PatchSize) * (cfg.ImageSize / cfg.PatchSize)
	x = Reshape(x, batchSize, numPatches, cfg.HiddenSize)
	
	posEmbed := ctx.VariableWithShape("position_embeddings", shapes.Make(x.DType(), numPatches, cfg.HiddenSize)).ValueGraph(g)
	x = Add(x, posEmbed)
	
	// 3. Token Reduction (Standard for Gemma 3: 4096 -> 256)
	// We assume a fixed reduction for now as seen in Gemma 3 specs
	if numPatches == 4096 {
		x = Reshape(x, batchSize, 16, 4, 16, 4, cfg.HiddenSize)
		x = ReduceMean(x, 2, 4)
		x = Reshape(x, batchSize, 256, cfg.HiddenSize)
	}
	
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("%d", i))
		x = SigLIPTransformerBlock(layerCtx, x, cfg.NumAttentionHeads, cfg.IntermediateSize)
	}
	
	return layers.LayerNormalization(ctx.In("final_norm"), x, -1).Done()
}
