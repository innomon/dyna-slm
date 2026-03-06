package gomlx_utils

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/nlpodyssey/safetensors"
)

// InitializeBackend sets up the XLA backend for GoMLX.
func InitializeBackend() (backends.Backend, error) {
	// Set the environment variable programmatically to hardcode XLA
	os.Setenv("GOMLX_BACKEND", "xla")

	// Updated v0.26.0 Signature: returns (Backend, error)
	backend, err := backends.New()
	if err != nil {
		return nil, fmt.Errorf("failed to create backend: %+v", err)
	}

	return backend, nil
}

// Model represents the T5Gemma 2 model context and weights.
type Model struct {
	Backend                backends.Backend
	Context                *context.Context
	Config                 *embedder.Config
	ExecEmbed              *graph.Exec
	ExecGenerate           *graph.Exec
	ExecDynaEncoder        *graph.Exec
	ExecDynaFusionDecoder  *graph.Exec
}

// NewModel initializes the GoMLX context.
func NewModel(backend backends.Backend, cfg *embedder.Config) *Model {
	return &Model{
		Backend: backend,
		Context: context.New(),
		Config:  cfg,
	}
}

// CompileEmbed compiles the multimodal embedding graph for inference.
func (m *Model) CompileEmbed(buildFn func(ctx *context.Context, textIds, imagePixels *graph.Node, cfg *embedder.Config) *graph.Node) {
	m.ExecEmbed = graph.MustNewExec(m.Backend, func(textIds, imagePixels *graph.Node) *graph.Node {
		return buildFn(m.Context, textIds, imagePixels, m.Config)
	})
}

// CompileGenerate compiles the multimodal generation graph.
func (m *Model) CompileGenerate(buildFn func(ctx *context.Context, textIds, imagePixels, decoderIds *graph.Node, cfg *embedder.Config) *graph.Node) {
	m.ExecGenerate = graph.MustNewExec(m.Backend, func(textIds, imagePixels, decoderIds *graph.Node) *graph.Node {
		return buildFn(m.Context, textIds, imagePixels, decoderIds, m.Config)
	})
}

// CompileDynaEncoder compiles the Dyna-SLM encoder graph.
func (m *Model) CompileDynaEncoder(buildFn func(ctx *context.Context, textIds, imagePixels *graph.Node, cfg *embedder.Config) []*graph.Node) {
	m.ExecDynaEncoder = graph.MustNewExec(m.Backend, func(textIds, imagePixels *graph.Node) []*graph.Node {
		return buildFn(m.Context, textIds, imagePixels, m.Config)
	})
}

// CompileDynaFusionDecoder compiles the Dyna-SLM fusion+decoder graph.
func (m *Model) CompileDynaFusionDecoder(buildFn func(ctx *context.Context, encoderHiddenStates, retrievedVectors, decoderIds *graph.Node, cfg *embedder.Config) *graph.Node) {
	m.ExecDynaFusionDecoder = graph.MustNewExec(m.Backend, func(encoderHiddenStates, retrievedVectors, decoderIds *graph.Node) *graph.Node {
		return buildFn(m.Context, encoderHiddenStates, retrievedVectors, decoderIds, m.Config)
	})
}

// DynaEncode executes the Dyna-SLM encoder graph.
func (m *Model) DynaEncode(textIds []uint32, imageTensor *tensors.Tensor) ([]float32, *tensors.Tensor, error) {
	if m.ExecDynaEncoder == nil {
		return nil, nil, fmt.Errorf("dyna encoder graph not compiled")
	}

	textT := tensors.FromFlatDataAndDimensions(textIds, 1, len(textIds))
	results := m.ExecDynaEncoder.MustExec(textT, imageTensor)
	
	pooledT := results[0]
	hiddenT := results[1]
	
	pooledData := pooledT.Value()
	var pooledVec []float32
	switch v := pooledData.(type) {
	case [][]float32:
		pooledVec = v[0]
	case []float32:
		pooledVec = v
	default:
		return nil, nil, fmt.Errorf("unexpected pooled tensor type: %T", pooledData)
	}
	
	return pooledVec, hiddenT, nil
}

// DynaFusionStep executes one step of the Dyna-SLM fusion+decoder graph.
func (m *Model) DynaFusionStep(encoderHiddenStates, retrievedVectors *tensors.Tensor, decoderIds []uint32) (uint32, error) {
	if m.ExecDynaFusionDecoder == nil {
		return 0, fmt.Errorf("dyna fusion decoder graph not compiled")
	}

	decoderT := tensors.FromFlatDataAndDimensions(decoderIds, 1, len(decoderIds))
	results := m.ExecDynaFusionDecoder.MustExec(encoderHiddenStates, retrievedVectors, decoderT)
	
	outT := results[0]
	data := outT.Value()
	
	var logits []float32
	switch v := data.(type) {
	case [][][]float32: // [batch][seq][vocab]
		batch := v[0]
		logits = batch[len(batch)-1]
	default:
		return 0, fmt.Errorf("unexpected logits tensor type: %T", data)
	}

	// Greedy: find max logit
	var maxIdx uint32
	var maxVal float32 = -1e10
	for i, val := range logits {
		if val > maxVal {
			maxVal = val
			maxIdx = uint32(i)
		}
	}

	return maxIdx, nil
}

// Embed executes the compiled GoMLX graph.
func (m *Model) Embed(textIds []uint32, imageTensor *tensors.Tensor) ([]float32, error) {
	if m.ExecEmbed == nil {
		return nil, fmt.Errorf("embedding graph not compiled")
	}

	textT := tensors.FromFlatDataAndDimensions(textIds, 1, len(textIds))

	results := m.ExecEmbed.MustExec(textT, imageTensor)
	if len(results) == 0 {
		return nil, fmt.Errorf("no output from graph execution")
	}

	outT := results[0]
	data := outT.Value()
	switch v := data.(type) {
	case [][]float32:
		return v[0], nil
	case []float32:
		return v, nil
	default:
		return nil, fmt.Errorf("unexpected tensor output type: %T", data)
	}
}

// GenerateStep executes one step of the generation graph.
func (m *Model) GenerateStep(textIds []uint32, imageTensor *tensors.Tensor, decoderIds []uint32) (uint32, error) {
	if m.ExecGenerate == nil {
		return 0, fmt.Errorf("generation graph not compiled")
	}

	textT := tensors.FromFlatDataAndDimensions(textIds, 1, len(textIds))
	decoderT := tensors.FromFlatDataAndDimensions(decoderIds, 1, len(decoderIds))

	results := m.ExecGenerate.MustExec(textT, imageTensor, decoderT)
	if len(results) == 0 {
		return 0, fmt.Errorf("no output from graph execution")
	}

	// Logits shape: [batch=1, seq_len, vocab_size]
	// We want the last token's prediction
	outT := results[0]
	data := outT.Value()
	
	var logits []float32
	switch v := data.(type) {
	case [][][]float32: // [batch][seq][vocab]
		batch := v[0]
		logits = batch[len(batch)-1]
	default:
		return 0, fmt.Errorf("unexpected logits tensor output type: %T", data)
	}

	// Greedy: find max logit
	var maxIdx uint32
	var maxVal float32 = -1e10
	for i, val := range logits {
		if val > maxVal {
			maxVal = val
			maxIdx = uint32(i)
		}
	}

	return maxIdx, nil
}

// LoadSafetensors loads one or more .safetensors files into the model's context.
func (m *Model) LoadSafetensors(weightsDir string) error {
	files, err := filepath.Glob(filepath.Join(weightsDir, "*.safetensors"))
	if err != nil {
		return fmt.Errorf("failed to list safetensors files: %w", err)
	}

	if len(files) == 0 {
		return fmt.Errorf("no .safetensors files found in %s", weightsDir)
	}

	for _, file := range files {
		fmt.Fprintf(os.Stderr, "📦 Loading weights from %s...\n", filepath.Base(file))
		
		data, err := os.ReadFile(file)
		if err != nil {
			return fmt.Errorf("failed to read safetensors file %s: %w", file, err)
		}

		st, err := safetensors.Deserialize(data)
		if err != nil {
			return fmt.Errorf("failed to deserialize safetensors from %s: %w", file, err)
		}

		for _, name := range st.Names() {
			t, ok := st.Tensor(name)
			if !ok {
				continue
			}

			gomlxScope := mapHuggingFaceToGoMLX(name)
			
			shape := make([]int, len(t.Shape()))
			for i, s := range t.Shape() {
				shape[i] = int(s)
			}

			var goMLXTensor *tensors.Tensor
			switch t.DType() {
			case safetensors.F32:
				tData := t.Data()
				floatData := *(*[]float32)(unsafe.Pointer(&tData))
				floatData = floatData[:len(tData)/4]
				goMLXTensor = tensors.FromFlatDataAndDimensions(floatData, shape...)
			default:
				continue
			}

			m.Context.In(gomlxScope).VariableWithShape("weight", shapes.Make(goMLXTensor.DType(), shape...)).MustSetValue(goMLXTensor)
		}
	}

	return nil
}

// mapHuggingFaceToGoMLX converts dot-separated HF names to GoMLX's slash-separated scopes.
func mapHuggingFaceToGoMLX(hfName string) string {
	name := strings.ReplaceAll(hfName, ".", "/")
	if strings.HasPrefix(hfName, "model.") {
		name = "/" + name[6:]
	} else if !strings.HasPrefix(hfName, "/") {
		name = "/" + name
	}
	return name
}
