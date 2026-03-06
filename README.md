# gomlx-pgvect-rag

A high-performance Multimodal Retrieval-Augmented Generation (RAG) system using **T5Gemma 2** (based on the **Gemma 3** architecture), **GoMLX** (XLA-accelerated), and **PostgreSQL** with **pgvector**. This project is packaged as a **Model Context Protocol (MCP) Server**.

## 🚀 Features
- **Multimodal Embedding:** Encodes both text and images into a shared vector space (640 or 768 dimensions).
- **Hybrid Architecture Support:** 
  - **T5Gemma 2:** SigLIP vision encoding + Gemma 3 transformer blocks.
  - **IBM Granite 4.0 350M-H:** Ultra-efficient Mamba-2 State Space Model (SSM) interleaved with GQA Transformer layers.
- **GoMLX Engine:** Pure Go implementation of the model graph, accelerated by XLA (CPU/GPU/TPU).
- **Scalable Vector Search:** High-performance similarity retrieval using pgvector HNSW indexing.
- **Dual Interface:** 
  - **MCP Native:** Integrated tool for AI clients (Goose, Claude, Gemini).
  - **OpenAI Compatible:** JWT-secured REST API for seamless integration with AnythingLLM, Goose, etc.

## 📋 Prerequisites
1. **Go 1.25+**
2. **PostgreSQL 16+** with the `pgvector` extension.
3. **libxla**: Required for GoMLX. Ensure `LD_LIBRARY_PATH` includes your XLA installation.
4. **CGO**: Required for building with the GoMLX/XLA backend.

## ⚙️ Setup

### 1. Database Configuration
Ensure PostgreSQL is running and initialize the schema:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(768)
);

CREATE INDEX ON filesys USING hnsw (embedding vector_cosine_ops);
```
Create a `.env` file in the root directory:
```bash
DATABASE_URL="postgres://user:pass@localhost:5432/dbname?sslmode=disable"
MODEL_WEIGHTS_DIR="./models/t5gemma-2-270m"
JWT_SECRET="your-secure-seed" # Seed for Ed25519 key generation
```

### 2. Download Model Weights
```bash
pip install huggingface_hub
huggingface-cli login
python3 download_weights.py
```

### 3. Build the Servers
```bash
# Build the MCP Server
CGO_ENABLED=1 go build -o mcp-server ./cmd/mcp-server

# Build the OpenAI-Compatible API Server
CGO_ENABLED=1 go build -o dyna-slm-api ./cmd/api
```

## 🛠️ Usage

### 1. OpenAI-Compatible API Provider
Run the API server:
```bash
./dyna-slm-api -weights ./models/t5gemma-2-270m -port 8080
```

#### Authentication (JWT with EdDSA)
The API is secured with JSON Web Tokens using the **EdDSA (Ed25519)** asymmetric algorithm.
- The server derives a stable Ed25519 keypair from the `JWT_SECRET` (seed).
- For testing, you can generate a token using:
```bash
curl http://localhost:8080/auth/token
```

#### Integration Example (Embeddings)
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"input": "Search query here", "model": "t5gemma-2-270m"}'
```

### 2. MCP Server
Add the following to your `goose` or `claude-desktop` configuration:
```yaml
extensions:
  gomlx-pgvect-rag:
    cmd: /path/to/gomlx-pgvect-rag/mcp-server
    args: ["-weights", "/path/to/gomlx-pgvect-rag/models/t5gemma-2-270m"]
    env:
      DATABASE_URL: "postgres://..."
```

### Available Tools
- `search_multimodal`: Search for assets using a text query or a local image path.
- `ingest_asset`: Ingest a file (image/text) to generate its embedding and store it in the database.

## 📂 Project Structure
- `cmd/api/`: OpenAI-compatible HTTP server and JWT auth.
- `cmd/mcp-server/`: Main entry point and MCP tool handlers.
- `internal/api/`: API types, handlers, and middleware.
- `internal/embedder/`: GoMLX implementation of SigLIP and Gemma 3 encoder blocks.
- `internal/db/`: PostgreSQL and pgvector persistence logic.
- `internal/gomlx_utils/`: Safetensors loading and XLA backend management.
- `pkg/utils/`: Pure Go tokenization and JWT security utilities.

## 📝 Troubleshooting
- **Authentication:** If you get SASL errors, verify your `DATABASE_URL` matches your PostgreSQL `pg_hba.conf` settings (use `peer` for local Linux users).
- **Logging:** The server redirects all status messages (e.g., "Loading weights") to `stderr` to avoid corrupting the MCP `stdout` stream.
