# UniContext Hierarchical Memory Skill

This skill enables agents to manage long-term memory and structured context using the **UniContext** mapping, a hierarchical filesystem-based approach for RAG.

## 1. Core Concepts

### A. Virtual Mapping
All hierarchical memory is stored in the `filesys` table with paths prefixed by `/unicontext/`.
- `viking://` URIs map to `/unicontext/` paths.
- Example: `viking://user/memories/` -> `/unicontext/user/memories/`

### B. Tiered Context (L0/L1/L2)
To maximize token efficiency, resources are split into three layers:
- **L0 (Abstract):** Short summary (~100 tokens). Path suffix: `/abstract`.
- **L1 (Overview):** Detailed overview (~2k tokens). Path suffix: `/overview`.
- **L2 (Details):** Full content. Path suffix: `/details`.

## 2. Agent Workflow

### A. Storing Memory
To "save" information into the hierarchical memory:
1.  **Prepare Content:** Create the L0, L1, and (optionally) L2 versions of the information.
2.  **Local File Creation:** Write the content to local files following the UniContext structure. 
    *Recommended Root:* `/home/innomon/AB2026Dev/dyna-slm/unicontext/`
3.  **Ingestion:** Use the `ingest_asset` tool for each layer.
    *   `ingest_asset(model="...", path="/path/to/local/unicontext/resource/abstract")`

### B. Recalling Memory (Scan-Before-Read)
To "recall" information efficiently:
1.  **Discovery (Scan L0):** Use `search_multimodal` with a query focused on the topic.
2.  **Filter:** From the results, identify assets with paths starting with `/unicontext/` and ending in `/abstract`.
3.  **Expansion (L1/L2):** If an abstract is highly relevant, fetch the corresponding `/overview` or `/details` by constructing the path and searching for it specifically (or using vector search with the abstract's path as a hint).

## 3. Tool Usage Patterns

### `search_multimodal`
- **Purpose:** Semantic discovery.
- **Instruction:** Always check the `path` of returned assets. Prioritize `/unicontext/` paths.
- **Efficiency:** If you find multiple layers for the same resource, start with the `/abstract`.

### `ingest_asset`
- **Purpose:** Committing to long-term memory.
- **Constraint:** The `path` provided must be a valid local file path. The agent should ensure the local directory structure mirrors the desired UniContext hierarchy.

## 4. Best Practices
- **Atomic Updates:** When updating a memory, ensure all layers (L0/L1) are updated to remain consistent.
- **Metadata usage:** Always include `{"layer": "L0", "viking_uri": "viking://..."}` in the metadata during ingestion if the tool allows (the current `ingest_asset` tool handles metadata internally but agents should be aware).
- **Context Management:** Do not load L2 (Details) into the conversation unless L1 (Overview) is insufficient to answer the user's query.
