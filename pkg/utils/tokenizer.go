package utils

import (
	"fmt"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Tokenizer wraps the sugarme/tokenizer.Tokenizer for easier use.
type Tokenizer struct {
	tk *tokenizer.Tokenizer
}

// NewTokenizer loads a tokenizer from a tokenizer.json file.
func NewTokenizer(path string) (*Tokenizer, error) {
	tk, err := pretrained.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %w", path, err)
	}
	return &Tokenizer{tk: tk}, nil
}

// Encode converts text into token IDs.
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) ([]uint32, error) {
	// sugarme/tokenizer uses a slightly different API
	en, err := t.tk.EncodeSingle(text, addSpecialTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}
	
	// Convert []int to []uint32
	ids := make([]uint32, len(en.Ids))
	for i, id := range en.Ids {
		ids[i] = uint32(id)
	}
	return ids, nil
}

// Close is a no-op for pure Go tokenizer.
func (t *Tokenizer) Close() {}
