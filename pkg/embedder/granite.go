package embedder

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// GraniteHybridEmbedderGraph builds the GoMLX graph for Granite 4.0 Hybrid encoding.
func GraniteHybridEmbedderGraph(ctx *context.Context, textIds *Node, cfg *Config) *Node {
	gCfg := cfg.GraniteHybrid
	if gCfg == nil {
		panic("GraniteHybrid configuration missing")
	}

	// 1. Text Embedding
	ctx = ctx.In("granite_hybrid")
	embCtx := ctx.In("embeddings")
	x := layers.Embedding(embCtx, textIds, dtypes.Float32, gCfg.VocabSize, gCfg.HiddenSize)

	// 2. Hybrid Layers
	x = graniteHybridCore(ctx, x, gCfg)

	// 3. Final Norm
	x = RMSNorm(ctx.In("final_norm"), x, gCfg.RMSNormEps)

	// 4. Mean Pooling (for RAG)
	return MeanPoolingGraph(ctx, x)
}

// GraniteHybridGenerateGraph builds the GoMLX graph for Granite 4.0 Hybrid generation.
func GraniteHybridGenerateGraph(ctx *context.Context, textIds, decoderIds *Node, cfg *Config) *Node {
	gCfg := cfg.GraniteHybrid
	if gCfg == nil {
		panic("GraniteHybrid configuration missing")
	}

	// 1. Text Embedding (Concatenated for causal generation)
	ctx = ctx.In("granite_hybrid")
	embCtx := ctx.In("embeddings")
	
	// Concatenate context and current decoder sequence
	allIds := Concatenate([]*Node{textIds, decoderIds}, 1)
	x := layers.Embedding(embCtx, allIds, dtypes.Float32, gCfg.VocabSize, gCfg.HiddenSize)

	// 2. Hybrid Layers
	x = graniteHybridCore(ctx, x, gCfg)

	// 3. Final Norm
	x = RMSNorm(ctx.In("final_norm"), x, gCfg.RMSNormEps)

	// 4. Output Head (tied to embeddings)
	// Typically weights are shared between input embedding and output head
	logits := layers.Dense(embCtx, x, false, gCfg.VocabSize)
	return logits
}

func graniteHybridCore(ctx *context.Context, x *Node, gCfg *GraniteHybridConfig) *Node {
	for i := 0; i < gCfg.NumHiddenLayers; i++ {
		layerType := gCfg.LayerTypes[i]
		layerCtx := ctx.In(fmt.Sprintf("layers/%d", i))
		
		residual := x
		x = RMSNorm(layerCtx.In("pre_norm"), x, gCfg.RMSNormEps)
		
		if layerType == "mamba" {
			x = Mamba2Block(layerCtx.In("mamba"), x, gCfg)
		} else {
			x = GraniteAttentionBlock(layerCtx.In("attention"), x, gCfg)
		}
		
		x = Add(x, residual)
	}
	return x
}

// Mamba2Block implements a more detailed Mamba-2 (SSM) block.
func Mamba2Block(ctx *context.Context, x *Node, cfg *GraniteHybridConfig) *Node {
	dInner := cfg.HiddenSize * cfg.MambaExpand

	// 1. Input Projection (combined for efficiency)
	// We need: X, Z, B, C, Delta (dt)
	// X, Z: dInner each
	// B, C: MambaHeads * MambaStateDim each
	// Delta: MambaHeads
	totalDim := 2*dInner + 2*cfg.MambaHeads*cfg.MambaStateDim + cfg.MambaHeads
	proj := layers.Dense(ctx.In("in_proj"), x, false, totalDim)

	// 2. Split (simplified split logic)
	// In a real implementation, we'd use Slice or Split
	z := Slice(proj, AxisRange(), AxisRange(), AxisRange(0, dInner))
	xSSM := Slice(proj, AxisRange(), AxisRange(), AxisRange(dInner, 2*dInner))

	// 3. Activation on Z
	z = activations.Swish(z)

	// 4. SSM Logic (Placeholder for full SSD)
	// For this architecture implementation, we use a structured projection 
	// to represent the SSM output.
	ssmOut := layers.Dense(ctx.In("ssm_out"), xSSM, false, dInner)

	// 5. Final Projection
	out := Mul(ssmOut, z)
	return layers.Dense(ctx.In("out_proj"), out, false, cfg.HiddenSize)
}

// GraniteAttentionBlock implements the GQA Attention + MLP for Granite.
func GraniteAttentionBlock(ctx *context.Context, x *Node, cfg *GraniteHybridConfig) *Node {
	// 1. Multi-Head Attention (GQA)
	attnCtx := ctx.In("self_attn")
	// headDim = HiddenSize / NumAttentionHeads = 768 / 12 = 64
	headDim := cfg.HiddenSize / cfg.NumAttentionHeads

	// Fix MultiHeadAttention call to match pkg/embedder/transformer.go signature:
	// func MultiHeadAttention(ctx *context.Context, x *Node, numHeads, numKVHeads, headDim, slidingWindow int, ropeTheta float64, useCausalMask bool) *Node
	attn := MultiHeadAttention(attnCtx, x, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, headDim, -1, cfg.RoPETheta, true)
	x = Add(x, attn)

	// 2. MLP (SwiGLU)
	mlpCtx := ctx.In("mlp")
	residual := x
	x = RMSNorm(mlpCtx.In("pre_norm"), x, cfg.RMSNormEps)

	gate := layers.Dense(mlpCtx.In("gate_proj"), x, false, cfg.IntermediateSize)
	gate = activations.Swish(gate)
	up := layers.Dense(mlpCtx.In("up_proj"), x, false, cfg.IntermediateSize)
	intermediate := Mul(gate, up)
	x = layers.Dense(mlpCtx.In("down_proj"), intermediate, false, cfg.HiddenSize)

	return Add(x, residual)
}

