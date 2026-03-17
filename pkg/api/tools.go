package api

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/innomon/gomlx-pgvect-rag/pkg/rag"
)

// Define the available tools matching the MCP server
var AvailableTools = []Tool{
	{
		Type: "function",
		Function: FunctionDefinition{
			Name:        "search_multimodal",
			Description: "Search for relevant text or image assets in the multimodal RAG store using a specific Dyna-SLM variant.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model": map[string]interface{}{
						"type":        "string",
						"description": "The Dyna-SLM model variant to use (e.g., dyna-gemma3-270m).",
					},
					"query_text": map[string]interface{}{
						"type":        "string",
						"description": "Textual query for similarity search.",
					},
					"query_image_path": map[string]interface{}{
						"type":        "string",
						"description": "Local path to an image file for visual similarity search.",
					},
					"limit": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of results to return (default: 5).",
					},
				},
				"required": []string{"model"},
			},
		},
	},
	{
		Type: "function",
		Function: FunctionDefinition{
			Name:        "ingest_asset",
			Description: "Ingest a new file (image or text) into the multimodal RAG store using a specific Dyna-SLM variant.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model": map[string]interface{}{
						"type":        "string",
						"description": "The Dyna-SLM model variant to use for embedding (e.g., dyna-gemma3-270m).",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Local path to the file to ingest.",
					},
				},
				"required": []string{"model", "path"},
			},
		},
	},
	{
		Type: "function",
		Function: FunctionDefinition{
			Name:        "list_variants",
			Description: "List all available Dyna-SLM model variants.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{},
			},
		},
	},
}

// ExecuteTool calls the appropriate orchestrator method based on the tool name and arguments
func ExecuteTool(ctx context.Context, registry *rag.Registry, name string, arguments string) (string, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %v", err)
	}

	switch name {
	case "search_multimodal":
		model, _ := args["model"].(string)
		if model == "" {
			return "", fmt.Errorf("missing required argument: model")
		}
		queryText, _ := args["query_text"].(string)
		queryImagePath, _ := args["query_image_path"].(string)
		limitF, ok := args["limit"].(float64)
		limit := 5
		if ok {
			limit = int(limitF)
		}

		orch, err := registry.GetOrchestrator(model)
		if err != nil {
			return "", err
		}

		assets, err := orch.Search(ctx, queryText, queryImagePath, limit)
		if err != nil {
			return "", err
		}

		resp, _ := json.Marshal(assets)
		return string(resp), nil

	case "ingest_asset":
		model, _ := args["model"].(string)
		path, _ := args["path"].(string)
		if model == "" || path == "" {
			return "", fmt.Errorf("missing required arguments: model, path")
		}

		orch, err := registry.GetOrchestrator(model)
		if err != nil {
			return "", err
		}

		err = orch.Ingest(ctx, path, map[string]interface{}{"source": "api-tool", "model": model})
		if err != nil {
			return "", err
		}

		return fmt.Sprintf("Successfully ingested into %s: %s", model, path), nil

	case "list_variants":
		variants := registry.ListModels()
		resp, _ := json.Marshal(variants)
		return string(resp), nil

	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}
