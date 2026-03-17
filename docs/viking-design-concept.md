# OpenViking Design Concept (Mapping-Based Adoption)

This document outlines the adoption of [OpenViking](https://github.com/volcengine/OpenViking) concepts into our Multimodal RAG system without modifying the existing codebase or database schema.

## 1. Core Paradigm: Virtual Mapping
Instead of a native protocol implementation, we map Viking filesystem concepts directly to the existing `filesys` table structure.

### A. Namespace Mapping
The `viking://` protocol is represented as a path prefix within the `path` column:
- `viking://` maps to `/unicontext/`
- `viking://resources/` -> `/unicontext/resources/`
- `viking://user/` -> `/unicontext/user/`
- `viking://agent/` -> `/unicontext/agent/`

### B. Tiered Context Loading (L0/L1/L2)
Layers are implemented through standardized path suffixes or metadata properties.

#### Path-Based Layering (Suffixes)
For a specific resource (e.g., `project_a/spec.md`), the layers are stored as separate entries:
- **L0 (Abstract):** `/unicontext/resources/project_a/spec.md/abstract`
- **L1 (Overview):** `/unicontext/resources/project_a/spec.md/overview`
- **L2 (Details):** `/unicontext/resources/project_a/spec.md/details`

#### Metadata-Based Layering
Alternatively, the `metadata` JSONB column distinguishes layers:
- `metadata -> 'layer'`: "L0", "L1", or "L2"
- `metadata -> 'source_path'`: The base logical path (e.g., `/unicontext/resources/project_a/spec.md`)

## 2. Implementation Strategy: Behavioral Patterns
Since no code changes are permitted, this design is enforced through **Agent Behavioral Instructions** (`AGENT.md` / `SKILL.md`). Agents interacting with the RAG system follow these patterns:

### A. Recursive Exploration Pattern
1. **Scan (L0):** Search for paths ending in `/abstract` or entries where `metadata @> '{"layer": "L0"}'`.
2. **Review (L1):** If the abstract is relevant, fetch the corresponding `/overview`.
3. **Deep Read (L2):** Fetch the `/details` (full content) only if the overview confirms necessity.

### B. Directory Navigation
Agents use SQL queries or existing tools to simulate filesystem operations:
- **List Directory:** `SELECT path FROM filesys WHERE path LIKE '/unicontext/resources/%' AND path NOT LIKE '%/%/%';`
- **Semantic Find:** Execute vector search but filter results to a specific "directory" using the `path` prefix.

## 3. Storage Compatibility
The existing `pkg/db/schema.sql` fully supports this mapping:
- `path TEXT PRIMARY KEY`: Handles hierarchical `/unicontext/` strings.
- `metadata JSONB`: Stores `layer` and `source_path` attributes for GIN-indexed filtering.
- `embedding vector(640)`: Allows for separate embeddings for abstracts (L0) and details (L2).

## 4. Operational Guidelines for Agents
Agents should prioritize the following when performing RAG:
1. **Efficiency:** Never fetch `/details` without first validating against the `/abstract`.
2. **Context Preservation:** When summarizing, always include the `source_path` to maintain the link to the hierarchical filesystem.
3. **Observability:** Log the "traversal path" (e.g., "Found relevant abstract at .../abstract, drilling down to .../details").
