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

// SigLIPVisionEncoder processes image tokens (27 layers).
func SigLIPVisionEncoder(ctx *context.Context, x *Node) *Node {
	ctx = ctx.In("siglip_vision")
	g := x.Graph()
	
	// 1. Patch Embedding (Convolution with stride 14)
	x = layers.Convolution(ctx.In("patch_embed"), x).
		Filters(1152).
		KernelSize(14).
		Strides(14).
		Done()
		
	// 2. Flatten and add Positional Embeddings
	batchSize := x.Shape().Dimensions[0]
	x = Reshape(x, batchSize, 4096, 1152)
	
	posEmbed := ctx.VariableWithShape("position_embeddings", shapes.Make(x.DType(), 4096, 1152)).ValueGraph(g)
	x = Add(x, posEmbed)
	
	// 3. Token Reduction: 4096 -> 256
	x = Reshape(x, batchSize, 16, 4, 16, 4, 1152)
	x = ReduceMean(x, 2, 4)
	x = Reshape(x, batchSize, 256, 1152)
	
	numLayers := 27
	numHeads := 16
	intermediateDim := 4304
	
	for i := 0; i < numLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("%d", i))
		x = SigLIPTransformerBlock(layerCtx, x, numHeads, intermediateDim)
	}
	
	return layers.LayerNormalization(ctx.In("final_norm"), x, -1).Done()
}
