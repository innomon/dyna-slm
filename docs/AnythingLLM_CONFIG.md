# [AnythinLLM MCP Server](https://docs.anythingllm.com/mcp-compatibility/desktop)

in the MCP config file : $HOME/.config/anythingllm-desktop/storage/plugins/anythingllm_mcp_servers.json
```json
{
  "mcpServers": {
    "gomlx-pgvect-rag": {
      "command": "$PORJECT_DIR/gomlx-pgvect-rag/mcp-server",
      "args": [],
      "env": {                          
        "DATABASE_URL": "postgres://user:pass@localhost:5432/postgres?sslmode=disable",
        "MODEL_WEIGHTS_DIR": "$PORJECT_DIR/gomlx-pgvect-rag/models/t5gemma-2-270m"
      }
    }
  }
}

```