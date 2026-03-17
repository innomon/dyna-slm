# GOOSE.md - UniContext Integration for Goose

This file provides specific instructions for Goose to operate the UniContext hierarchical memory architecture within the `gomlx-pgvect-rag` workspace.

## 1. Role and Mandate
Goose MUST follow the hierarchical context pattern described in `AGENT.md`. This pattern allows Goose to manage long-term memory efficiently by "scanning before reading."

## 2. Core Concepts
- **Virtual Protocol:** Map `viking://` to the `/unicontext/` path prefix.
- **Layering:** 
  - **L0 (Abstract):** `.../abstract` (Scan this first).
  - **L1 (Overview):** `.../overview` (Read to confirm relevance).
  - **L2 (Details):** `.../details` (Deep dive only if needed).

## 3. Tool Execution Patterns

### Discovery (Scan L0)
When asked for information, Goose should first use:
```bash
search_multimodal(query_text="relevant query terms", model="dyna-gemma3-270m", limit=5)
```
Then filter the results to find paths ending in `/abstract`.

### Ingestion (Tiered Storage)
When Goose needs to "remember" something:
1.  **Summarize:** Create an L0 abstract and an L1 overview.
2.  **Save Local:** Save these to `unicontext/resources/.../abstract` and `unicontext/resources/.../overview`.
3.  **Ingest:** Call `ingest_asset` for each layer.

## 4. Integration Details
- Refer to `docs/GOOSE_INTEGRATION.md` for mcp-server setup instructions.
- Goose is configured to use the `mcp-server` in this project as a primary memory tool provider.
