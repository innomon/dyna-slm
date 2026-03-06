package api

// OpenAI Request Types
type ChatCompletionRequest struct {
	Model    string                  `json:"model"`
	Messages []ChatCompletionMessage `json:"messages"`
	Stream   bool                    `json:"stream"`
}

type ChatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type EmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // Can be string or []string
}

// OpenAI Response Types
type ChatCompletionResponse struct {
	Id      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []ChatCompletionChoice   `json:"choices"`
}

type ChatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type ModelList struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}
