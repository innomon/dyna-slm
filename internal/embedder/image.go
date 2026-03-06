package embedder

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/disintegration/imaging"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

const (
	// SigLIP Standard Resolution
	ImageSize = 896
)

// LoadImageAsTensor reads an image from disk, resizes it to 896x896,
// and returns a GoMLX tensor of shape [1, 896, 896, 3] in RGB.
func LoadImageAsTensor(path string) (*tensors.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	resized := imaging.Resize(img, ImageSize, ImageSize, imaging.Lanczos)

	bounds := resized.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	data := make([]float32, h*w*3)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			offset := (y*w + x) * 3
			data[offset] = float32(r >> 8)
			data[offset+1] = float32(g >> 8)
			data[offset+2] = float32(b >> 8)
		}
	}

	t := tensors.FromFlatDataAndDimensions(data, 1, h, w, 3)
	return t, nil
}

// LoadImageFromBytes decodes an image from memory, resizes it, and returns a GoMLX tensor.
func LoadImageFromBytes(data []byte) (*tensors.Tensor, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image from bytes: %w", err)
	}

	resized := imaging.Resize(img, ImageSize, ImageSize, imaging.Lanczos)
	bounds := resized.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	flatData := make([]float32, h*w*3)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			offset := (y*w + x) * 3
			flatData[offset] = float32(r >> 8)
			flatData[offset+1] = float32(g >> 8)
			flatData[offset+2] = float32(b >> 8)
		}
	}

	t := tensors.FromFlatDataAndDimensions(flatData, 1, h, w, 3)
	return t, nil
}

// IsImage simple check for image content based on magic numbers.
func IsImage(data []byte) bool {
	if len(data) < 4 {
		return false
	}
	// JPEG
	if data[0] == 0xff && data[1] == 0xd8 {
		return true
	}
	// PNG
	if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4e && data[3] == 0x47 {
		return true
	}
	return false
}
