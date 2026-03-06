# Goose Integration Guide

To use `gomlx-pgvect-rag` with [Goose](https://block.github.io/goose/), you need to register it as an extension. This allows Goose to use the T5Gemma 2 model to search and ingest your local files.

## 1. Locate your Goose Configuration
Typically located at:
- **Linux/macOS:** `~/.config/goose/config.yaml`
- **Windows:** `%APPDATA%\goose\config.yaml`

## 2. Add the Extension
Add the following entry to the `extensions` section of your `config.yaml`. 

**Note:** Replace `/home/innomon/orez/gomlx-pgvect-rag` with the absolute path to your project directory.

```yaml
extensions:
  gomlx-pgvect-rag:
    cmd: /home/innomon/orez/gomlx-pgvect-rag/mcp-server
    args: 
      - "-weights"
      - "/home/innomon/orez/gomlx-pgvect-rag/models/t5gemma-2-270m"
    env:
      DATABASE_URL: "postgres://innomon@localhost:5432/postgres?sslmode=disable"
```

## 3. Verify the Setup
Restart Goose and look for the successful initialization message in your terminal:
```text
📂 Loading weights from: /home/innomon/orez/gomlx-pgvect-rag/models/t5gemma-2-270m
🛠️  Compiling GoMLX graph...
🚀 gomlx-pgvect-rag MCP Server running on xla:0
```

## 4. Usage Example
Once integrated, you can simply tell Goose:
> "Hey Goose, ingest the image at ./my_photo.jpg into my RAG store."

Goose will automatically call the `ingest_asset` tool provided by this server.
