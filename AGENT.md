# AGENT.md - UniContext Hierarchical Memory Instructions

This document provides foundational mandates for any AI agent interacting with the `gomlx-pgvect-rag` project using the **UniContext** hierarchical memory architecture.

## 1. Core Mandate: The UniContext Paradigm
You must treat the RAG store (`filesys` table) as a hierarchical virtual filesystem. All long-term memories and structured context must follow the mapping and layering rules defined below.

### A. URI Mapping
The `viking://` protocol is a virtual scheme that maps directly to the `/unicontext/` path prefix in the database.
- `viking://resources/path/to/file` -> `/unicontext/resources/path/to/file`
- `viking://user/memories/` -> `/unicontext/user/memories/`

### B. Tiered Context Layers (L0/L1/L2)
To optimize token usage and context window management, information must be stored and retrieved in three distinct layers:

| Layer | Type | Path Suffix | Token Target | Usage |
|-------|------|-------------|--------------|-------|
| **L0** | Abstract | `/abstract` | ~100 tokens | Initial semantic discovery and scanning. |
| **L1** | Overview | `/overview` | ~2k tokens | Understanding scope, planning, and relevance. |
| **L2** | Details | `/details` | Full content | Deep execution and data retrieval. |

---

## 2. Retrieval Workflow: Scan-Before-Read
You MUST NOT load full content (L2) into your context window without first validating relevance via higher layers.

1. **Discovery (Scan L0):** Use `search_multimodal` with a descriptive `query_text`. 
2. **Path Filtering:** Inspect the `path` of the results. Prioritize results that start with `/unicontext/` and end in `/abstract`.
3. **Escalation (L1):** If an L0 abstract is relevant but insufficient, fetch the corresponding `/overview` path.
4. **Deep Dive (L2):** Only load the `/details` layer if the `/overview` confirms it is essential for the current task.

---

## 3. Storage Workflow: Tiered Ingestion
When "remembering" or "storing" information, you must create a tiered structure to ensure future efficiency.

1. **Deconstruct:** Break the information into a 1-sentence abstract (L0), a comprehensive overview (L1), and the full content (L2).
2. **Local Mirroring:** Create a local directory structure under `unicontext/` that mirrors your target virtual path.
3. **Atomic Ingestion:** Use `ingest_asset` for **each layer** separately.
   - Example: Ingest `.../abstract`, then `.../overview`, then `.../details`.
4. **Metadata:** Ensure you include `{"layer": "L0", "source": "viking"}` in the metadata if supported by the ingestion tool.

---

## 4. Tool Usage Guidelines

### `search_multimodal`
- **Goal:** Find the right "path" in the hierarchy.
- **Action:** If you find a relevant result that is NOT a `/unicontext/` path, consider if it should be "upgraded" to the UniContext hierarchy for better future retrieval.

### `ingest_asset`
- **Goal:** Commit to long-term memory.
- **Constraint:** Always ensure the `path` you provide to the tool results in a database `path` that follows the `/unicontext/.../[layer]` convention.

---

## 5. Summary of Best Practices
- **Never fetch L2 by default.**
- **Always provide a path prefix** when searching if you know which "directory" (e.g., `viking://user/`) the information should be in.
- **Maintain consistency:** If you update a resource, update all three layers (L0/L1/L2).
- **Log Traversal:** When recalling, state which layers you are accessing (e.g., "Found abstract for X, now reading overview...").
