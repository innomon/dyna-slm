package embedder

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// RMSNorm implements Root Mean Square Layer Normalization.
func RMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	ctx = ctx.In("rms_norm")
	g := x.Graph()
	ms := ReduceMean(Square(x), -1)
	invRms := Inverse(Sqrt(AddScalar(ms, epsilon)))
	normalized := Mul(x, invRms)
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	gamma := ctx.VariableWithShape("weight", shapes.Make(x.DType(), hiddenDim)).ValueGraph(g)
	return Mul(normalized, gamma)
}

// MLP block for Gemma 3.
func MLP(ctx *context.Context, x *Node, intermediateDim int) *Node {
	ctx = ctx.In("mlp")
	gate := layers.Dense(ctx.In("gate_proj"), x, true, intermediateDim)
	gate = activations.Gelu(gate)
	up := layers.Dense(ctx.In("up_proj"), x, true, intermediateDim)
	intermediate := Mul(gate, up)
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	return layers.Dense(ctx.In("down_proj"), intermediate, true, hiddenDim)
}

// MultiHeadAttention implements the Gemma 3 attention mechanism.
func MultiHeadAttention(ctx *context.Context, x *Node, numHeads, numKVHeads, headDim, slidingWindow int, ropeTheta float64) *Node {
	ctx = ctx.In("attention")
	return layers.MultiHeadAttention(ctx, x, x, x, numHeads, numHeads*headDim).Done()
}

// EncoderBlock represents a single Gemma 3 transformer layer.
func EncoderBlock(ctx *context.Context, x *Node, numHeads, numKVHeads, headDim, intermediateDim, slidingWindow int, ropeTheta float64) *Node {
	normX := RMSNorm(ctx.In("pre_attention_norm"), x, 1e-6)
	attn := MultiHeadAttention(ctx, normX, numHeads, numKVHeads, headDim, slidingWindow, ropeTheta)
	x = Add(x, attn)
	normX = RMSNorm(ctx.In("pre_mlp_norm"), x, 1e-6)
	mlpOut := MLP(ctx, normX, intermediateDim)
	x = Add(x, mlpOut)
	return x
}

// Gemma3Encoder assembles 18 transformer layers for T5Gemma 2-270M.
func Gemma3Encoder(ctx *context.Context, x *Node) *Node {
	ctx = ctx.In("gemma3_encoder")
	numLayers := 18
	numHeads := 4
	numKVHeads := 1
	headDim := 256
	intermediateDim := 2048
	slidingWindow := 512

	for i := 0; i < numLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("%d", i))
		ropeTheta := 10000.0
		currentSlidingWindow := slidingWindow
		if (i+1)%6 == 0 {
			ropeTheta = 1000000.0
			currentSlidingWindow = -1
		}
		x = EncoderBlock(layerCtx, x, numHeads, numKVHeads, headDim, intermediateDim, currentSlidingWindow, ropeTheta)
	}
	return RMSNorm(ctx.In("final_norm"), x, 1e-6)
}
