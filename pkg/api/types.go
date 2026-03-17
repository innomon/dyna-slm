package api

// OpenAI Request Types
type ChatCompletionRequest struct {
	Model      string                  `json:"model"`
	Messages   []ChatCompletionMessage `json:"messages"`
	Stream     bool                    `json:"stream"`
	Tools      []Tool                  `json:"tools,omitempty"`
	ToolChoice interface{}             `json:"tool_choice,omitempty"` // "none", "auto", "required", or ToolChoice object
}

type ChatCompletionMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallId string     `json:"tool_call_id,omitempty"` // For role: "tool"
}

type Tool struct {
	Type     string             `json:"type"` // always "function"
	Function FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"` // JSON Schema
}

type ToolCall struct {
	Id       string           `json:"id"`
	Type     string           `json:"type"` // always "function"
	Function FunctionInstance `json:"function"`
}

type FunctionInstance struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

type ToolChoice struct {
	Type     string               `json:"type"`
	Function ToolChoiceFunction `json:"function"`
}

type ToolChoiceFunction struct {
	Name string `json:"name"`
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

// Responses API Types (New)
type ResponseRequest struct {
	Model          string      `json:"model"`
	Input          interface{} `json:"input"` // Can be string or []ResponseItem
	Instructions   string      `json:"instructions,omitempty"`
	Store          bool        `json:"store,omitempty"`
	ConversationId string      `json:"conversation_id,omitempty"`
	Metadata       interface{} `json:"metadata,omitempty"`
	Stream         bool        `json:"stream,omitempty"`
	Tools          []Tool      `json:"tools,omitempty"`
	ToolChoice     interface{} `json:"tool_choice,omitempty"`
}

type ResponseItem struct {
	Id      string            `json:"id,omitempty"`
	Type    string            `json:"type"` // "message", "function_call", "function_call_output"
	Role    string            `json:"role,omitempty"`
	Content []ResponseContent `json:"content,omitempty"`
	Call    *ToolCall         `json:"call,omitempty"`
	Output  string            `json:"output,omitempty"`
}

type ResponseContent struct {
	Type string `json:"type"` // "text", "image_url"
	Text string `json:"text,omitempty"`
}

type ResponseResponse struct {
	Id             string         `json:"id"`
	Object         string         `json:"object"`
	Created        int64          `json:"created"`
	Model          string         `json:"model"`
	Output         []ResponseItem `json:"output"`
	ConversationId string         `json:"conversation_id,omitempty"`
	Usage          Usage          `json:"usage,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
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
