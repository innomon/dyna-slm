# DYNA-SLM Specification: Dynamic Data Tuned Small Language Model

This document specifies the architecture and design of the **Dyna-SLM** (Dynamic Data Tuned Small Language Model) system, featuring a deep-integrated Retrieval-Augmented Generation (RAG) layer.

## 1. Architectural Overview

Dyna-SLM differs from traditional RAG by integrating the retrieval step directly into the model's inference pipeline between the encoder and decoder stages.

### 1.1. Inference Pipeline (Embedded RAG Layer)
1.  **Input Encoding:** The user chat query is processed through initial transformer layers up to the **Encoder Layer**.
2.  **Latent Retrieval:** The output embedding from the encoder is extracted. This embedding serves as the query vector for the vector database.
3.  **Vector Search:** A configurable number of records (`k`) are retrieved from the database associated with the model's specific embedding dimension and architecture.
4.  **Context Integration:** The encoded user input (latent states) and the retrieved vectors (latent states of retrieved records) are passed through an intermediate **Fusion Transformer**.
5.  **Decoding:** The fused representation is passed to the **Decoder Layer** for final token generation.
6.  **Reference Appending:** The file paths or identifiers of the retrieved records are maintained separately and appended to the final generated output as "References".

## 2. Dynamic Model Variants

Dyna-SLM supports multiple model variants defined by a configuration file. Each variant is treated as a distinct model in the OpenAI-compatible API.

### 2.1. Configuration Parameters
Each model is defined by:
-   **Model Name:** The identifier used in the API (e.g., `dyna-gemma-3-small`).
-   **Architecture:** The base model architecture (currently supported: `gemma3`, `granite-hybrid`).
-   **Weights Location:** Absolute or relative path to the model weights file (.bin, .safetensors).
-   **Embedding Dimensions:** The hidden size of the encoder/decoder (e.g., 640 for Gemma 3 270M, 768 for Granite 350M).
-   **Database Name:** The specific PostgreSQL table or database instance to use for retrieval.
-   **Pre-filter SQL:** An optional SQL snippet (e.g., `metadata->'category' = 'legal'`) to filter records before vector search.

### 2.2. Vector Database Strategy
-   **Dimension-Specific Storage:** Databases are partitioned by embedding dimension to ensure compatibility between the model's latent space and the stored vectors.
-   **Algorithm Alignment:** Each database is optimized for the specific embedding algorithm used by its corresponding encoder.
- **Filtered Search:** The retrieval step applies the **Pre-filter SQL** to the `filesys` table (filtering by metadata or sub-path) before executing the vector similarity search (`<=>`). This allows model customizers to restrict the knowledge base for specific model variants.

### 2.3. Migration and Re-embedding
Dyna-SLM includes a migration utility to support architecture transitions:
-   **Model Switching:** When moving from one model (e.g., Gemma 3) to another (e.g., Granite), the system re-embeds all assets using the target model's encoder.
-   **Metadata Modification:** Allows bulk addition or removal of JSONB metadata fields during migration.
-   **Batch Processing:** Uses paginated retrieval to migrate large datasets efficiently.


## 3. MCP Server Integration

The system exposes specialized **MCP Servers** based on the model configuration.

-   **Specialization:** MCP instances are launched with specific configurations: `encoder`, `vector dimension`, and `database name`.
-   **Tools:**
    -   `dyna_search`: Perform the embedded retrieval step.
    -   `dyna_ingest`: Process and store data into the dimension-specific database.
    -   `dyna_list_variants`: List available model configurations.

## 4. API Compatibility

Dyna-SLM implements an OpenAI-compatible interface:
-   **`GET /v1/models`**: Returns the list of configured Dyna-SLM variants.
-   **`POST /v1/chat/completions`**: Executes the full Dyna-SLM inference pipeline, including the embedded RAG layer and reference appending.

## 5. Technical Stack
-   **ML Engine:** GoMLX (XLA-accelerated).
-   **Vector Store:** PostgreSQL + `pgvector`.
-   **Orchestration:** Go (standard library + MCP SDK).
-   **Tokenizer:** Architecture-specific (Gemma 3 / Granite).
