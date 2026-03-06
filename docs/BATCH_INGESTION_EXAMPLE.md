# Batch Ingestion Example

To ingest all images or documents from a directory at once, you can use a simple script that interacts with the `mcp-server`. While MCP is designed for LLM interaction, you can also use a helper script to bulk-load data.

## Using a Bash Script
This script loops through all `.jpg` files in a directory and uses an MCP tool-caller (like `mcp-cli` or a custom Go helper) to ingest them.

```bash
#!/bin/bash

# Directory containing your images
DATA_DIR="./my_images"

# 1. Start your MCP server in the background (if not already running via Goose)
# ./mcp-server -weights ./models/t5gemma-2-270m &

# 2. Iterate and Ingest
for file in "$DATA_DIR"/*.jpg; do
    echo "📥 Ingesting: $file"
    # Using Goose or an MCP CLI to call the tool
    goose session "ingest the file at $file into gomlx-pgvect-rag"
done
```

## Prompt-based Batch Ingestion
You can also simply tell Goose to do the heavy lifting for you:

> "Hey Goose, find all JPG files in the directory './knowledge_base' and ingest each of them using the 'ingest_asset' tool."

Goose will then:
1. List the directory.
2. Filter for JPG files.
3. Call `ingest_asset` for every file it found.

## Performance Tip
If you are ingesting thousands of files, it is more efficient to use a Go script that directly calls `internal/rag/orchestrator.go`'s `Ingest` method to avoid the JSON-RPC overhead of individual tool calls.
