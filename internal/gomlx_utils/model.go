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
	"github.com/nlpodyssey/safetensors"
)

// InitializeBackend sets up the XLA backend for GoMLX.
func InitializeBackend() (backends.Backend, error) {
	if os.Getenv("GOMLX_BACKEND") == "" {
		os.Setenv("GOMLX_BACKEND", "xla")
	}

	backend, err := backends.New()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GoMLX backend: %w", err)
	}

	return backend, nil
}

// Model represents the T5Gemma 2 model context and weights.
type Model struct {
	Backend   backends.Backend
	Context   *context.Context
	ExecEmbed *graph.Exec
}

// NewModel initializes the GoMLX context.
func NewModel(backend backends.Backend) *Model {
	return &Model{
		Backend: backend,
		Context: context.New(),
	}
}

// CompileEmbed compiles the multimodal embedding graph for inference.
func (m *Model) CompileEmbed(buildFn func(ctx *context.Context, textIds, imagePixels *graph.Node) *graph.Node) {
	m.ExecEmbed = graph.MustNewExec(m.Backend, func(textIds, imagePixels *graph.Node) *graph.Node {
		return buildFn(m.Context, textIds, imagePixels)
	})
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
	// Using Value() to get the actual underlying slice (copy)
	data := outT.Value()
	// tensors.Value() returns []T or [][]T etc. for rank > 0.
	// For Rank 2 [1, 640], it returns [][]float32.
	// We want flat float32.
	switch v := data.(type) {
	case [][]float32:
		return v[0], nil
	case []float32:
		return v, nil
	default:
		return nil, fmt.Errorf("unexpected tensor output type: %T", data)
	}
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
