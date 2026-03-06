# Dyna-SLM: Deep Latent Retrieval and Fusion for Dynamic Contextual Adaptation in Small Language Models

**Abstract**
Small Language Models (SLMs) offer significant advantages in edge deployment and inference efficiency but often suffer from restricted internal knowledge bases and a tendency toward hallucination. We present **Dyna-SLM (Dynamic Data Tuned Small Language Model)**, an architecture that internalizes the Retrieval-Augmented Generation (RAG) process. Unlike traditional RAG, which operates at the text-to-text interface, Dyna-SLM integrates retrieval directly into the model’s latent space between the encoder and decoder stages. By employing a Fusion Transformer to synthesize encoder hidden states with retrieved latent vectors, Dyna-SLM achieves a higher degree of contextual grounding. We demonstrate that this approach provides a more scalable, verifiable, and computationally efficient alternative to frequent model fine-tuning.

---

## 1. Introduction
The current paradigm of adapting Large Language Models (LLMs) to specific domains relies heavily on either Parameter-Efficient Fine-Tuning (PEFT) or post-hoc Retrieval-Augmented Generation (RAG). While fine-tuning embeds knowledge into the model's weights, it is static and prone to catastrophic forgetting. Traditional RAG, conversely, provides dynamic updates but introduces a "semantic bottleneck" by forcing the model to interpret retrieved data only through the lens of natural language.

**Dyna-SLM** bridges this gap by embedding the RAG layer within the transformer architecture itself. By extracting latent representations mid-inference and using them to query dimension-aligned vector databases, the model can "perceive" external data as native latent states rather than just additional input text.

## 2. Architecture: The Embedded RAG Layer
The Dyna-SLM pipeline consists of four distinct phases executed within a single GoMLX-accelerated graph:

### 2.1 Latent Encoding
The user query (multimodal or text-only) is processed through the initial transformer layers (based on Gemma 3 or Granite architectures). At the terminal layer of the encoder, we extract two outputs:
1. **Full Hidden States:** The sequence-length tensor of latent representations.
2. **Mean-Pooled Query Vector:** A compressed latent representation used for vector database indexing.

### 2.2 Domain-Filtered Vector Search
Dyna-SLM utilizes a PostgreSQL `pgvector` store partitioned by embedding dimension (e.g., 640-dim for Gemma 3 270M, 768-dim for Granite 350M). Innovation in this layer includes a **SQL Pre-filter**, allowing model customizers to restrict the search space based on metadata or sub-paths (e.g., `path LIKE '/legal/%'`) *before* the similarity search is executed.

### 2.3 The Fusion Transformer
The retrieved records are not converted back to text immediately. Instead, their stored latent vectors are fed into a **Fusion Transformer**. This module employs cross-attention where the encoder hidden states act as queries and the retrieved vectors act as keys and values. This allows the model to "tune" its internal representation of the query based on the retrieved context before generation begins.

### 2.4 Reference-Aware Decoding
The fused representation is passed to the decoder. Throughout this process, the source identifiers (file paths) of the retrieved vectors are maintained in a parallel metadata buffer. These are appended to the final token output as verifiable "References," ensuring transparency and reducing hallucination.

## 3. Innovation: Latent-to-Latent Interaction
The primary innovation of Dyna-SLM is the removal of the text-based bottleneck. In standard RAG, retrieved text must be re-tokenized and re-encoded, losing the original semantic nuances captured during the initial ingestion. Dyna-SLM retrieves the **original latent vectors**, ensuring that the context integration happens at the same "mathematical resolution" as the model's own internal reasoning.

## 4. Comparative Analysis: Dyna-SLM vs. Fine-Tuning

| Feature | Fine-Tuning (PEFT/LoRA) | Dyna-SLM (Embedded RAG) |
| :--- | :--- | :--- |
| **Knowledge Latency** | Static (requires re-training) | Real-time (update DB, model adapts) |
| **Reliability** | Prone to "Hallucination" | Verifiable via Reference Appending |
| **Computational Cost** | High (GPU hours for training) | Low (Inference-time DB query) |
| **Data Privacy** | Hard to "Unlearn" specific data | Easy (Delete record from DB) |
| **Context Window** | Limited by sequence length | Scalable via K-record retrieval |
| **Architectural Depth** | Modifies existing weights | Adds a specialized Fusion Layer |

### 4.1 Stability and Forgetting
Fine-tuning often results in the model losing general-purpose capabilities (catastrophic forgetting). Dyna-SLM maintains the integrity of the foundation model's weights while using the Fusion Transformer to provide a "temporary" tuning specialized for the current query.

### 4.2 Verifiability
A fine-tuned model provides answers from its weights but cannot cite its sources. Dyna-SLM provides a direct trace from the generated token back to the specific vector record in the database, satisfying requirements for auditability in enterprise environments.

## 5. Conclusion
Dyna-SLM represents a shift from "Models that Know" to "Models that Reason over Data." By internalizing retrieval into the latent space and employing a fusion-based architecture, Dyna-SLM provides the performance of a fine-tuned domain-specific model with the flexibility and verifiability of a RAG system. This architecture is particularly suited for Small Language Models, where parameter count is precious and must be augmented by external, dynamic memory.

---
**Keywords:** Small Language Models, RAG, GoMLX, Latent Retrieval, Fusion Transformer, Gemma 3, pgvector.
