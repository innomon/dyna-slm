# Dyna-SLM: Dynamic Data Tuned Small Language Model

A high-performance Multimodal Retrieval-Augmented Generation (RAG) system with a deep-integrated retrieval layer. Dyna-SLM uses **T5Gemma 2** (Gemma 3 architecture) and **IBM Granite 4.0**, accelerated by **GoMLX/XLA**, and backed by **PostgreSQL** with **pgvector**.

## 🚀 Features
- **Deep Integrated RAG (Embedded Layer):** Retrieval occurs *within* the model's latent space. Encoder latent states are fused with retrieved vectors via a specialized **Fusion Transformer** before decoding.
- **Latent-to-Latent Interaction:** Bypasses the text bottleneck by retrieving and fusing raw latent vectors, ensuring higher semantic fidelity.
- **Multimodal Support:** Encodes and reasons over both text and images (SigLIP vision encoder).
- **Hybrid Architecture Support:** 
  - **T5Gemma 2:** SigLIP + Gemma 3 transformer blocks.
  - **IBM Granite 4.0 350M-H:** Mamba-2 SSM interleaved with GQA Transformer layers.
- **Dynamic Model Variants:** Configure multiple model variants (different weights, architectures, dimensions) in a single instance.
- **SQL Pre-filtering:** Custom model-level knowledge gating using SQL `WHERE` clauses (e.g., metadata filters or path restrictions).
- **Verifiable Output:** Automatically appends source references to the generated response.
- **Dual Interface:** 
  - **MCP Native:** Tools for AI clients (Goose, Claude, Gemini).
  - **OpenAI Compatible:** JWT-secured REST API (Ed25519) for AnythingLLM, Goose, etc.

## 📋 Prerequisites
1. **Go 1.25+**
2. **PostgreSQL 16+** with the `pgvector` extension.
3. **libxla**: Required for GoMLX. Ensure `LD_LIBRARY_PATH` includes your XLA installation.
4. **CGO**: Required for building with the GoMLX/XLA backend.

## ⚙️ Setup

### 1. Database Configuration
Ensure PostgreSQL is running. Dyna-SLM automatically initializes dimension-specific tables (e.g., `filesys_640`) based on your configuration.

Create a `.env` file:
```bash
DATABASE_URL="postgres://user:pass@localhost:5432/dbname?sslmode=disable"
JWT_SECRET="your-secure-seed" # Seed for Ed25519 key derivation
```

### 2. Model Configuration (`models.json`)
Define your model variants in a JSON file:
```json
[
  {
    "name": "dyna-gemma3-270m",
    "architecture": "gemma3",
    "weights_path": "./models/t5gemma-2-270m/model.bin",
    "embedding_dimension": 640,
    "database_name": "filesys_640",
    "pre_filter_sql": "path LIKE '/docs/public/%'",
    "k": 5
  }
]
```

### 3. Build the Servers
```bash
# Build the MCP Server
CGO_ENABLED=1 go build -o dyna-mcp ./cmd/mcp-server

# Build the OpenAI-Compatible API Server
CGO_ENABLED=1 go build -o dyna-api ./cmd/api
```

## 🛠️ Usage

### 1. OpenAI-Compatible API Provider
Run the API server:
```bash
./dyna-api -config models.json -port 8080
```
- **Authentication:** Derives a stable Ed25519 keypair from `JWT_SECRET`. Generate a test token at `GET /auth/token`.
- **Endpoints:** Supports standard `/v1/chat/completions`, `/v1/embeddings`, and `/v1/models`.

### 2. MCP Server
Add to your `goose` or `claude-desktop` config:
```yaml
extensions:
  dyna-slm:
    cmd: /path/to/dyna-slm/dyna-mcp
    args: ["-config", "/path/to/models.json"]
    env:
      DATABASE_URL: "postgres://..."
```

### Available Tools
- `search_multimodal`: Search assets using a text query or local image path.
- `ingest_asset`: Ingest a file and generate its dimension-specific embedding.
- `list_variants`: List available configured model variants.

## 📖 Documentation
- **Architecture & Innovation:** See [DYNA_SLM_PAPER.md](./DYNA_SLM_PAPER.md) for a technical breakdown of latent retrieval vs. fine-tuning.
- **Specification:** Detailed system design in [DYNA_SLM_SPECIFICATION.md](./DYNA_SLM_SPECIFICATION.md).
- **Implementation Plan:** Track progress in [DYNA_SLM_IMPLEMENTATION_PLAN.md](./DYNA_SLM_IMPLEMENTATION_PLAN.md).

## 📄 License
Licensed under the [Apache License, Version 2.0](./LICENSE).
