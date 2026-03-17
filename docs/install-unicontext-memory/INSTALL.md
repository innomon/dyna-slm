# Installing UniContext for Dyna-SLM

Provide long-term memory capabilities for Dyna-SLM via the **UniContext** mapping. This approach uses the OpenViking hierarchical context paradigm to organize information into tiered layers (L0/L1/L2).

---

## One-Click Setup (Conceptual)

Since UniContext is a mapping pattern for the existing `gomlx-pgvect-rag` project, you don't need to install a separate "plugin." You simply apply the naming conventions to your data ingestion.

To "install" the UniContext structure:
1. **Prepare Root:** Create `/home/innomon/AB2026Dev/dyna-slm/unicontext/`.
2. **Apply Skill:** Add `docs/install-unicontext-memory/SKILL.md` to your agent's system prompt.
3. **Run Migration:** Ensure your PostgreSQL table has the correct indices (already present in `pkg/db/schema.sql`).

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| **Go** | >= 1.23 | Backend logic |
| **Postgres** | >= 15 | Vector Database |
| **Gemma 3** | 270M | Embedding Model |

Check:
```bash
go version
psql --version
# Verify the dyna-slm API is running
curl -X POST http://localhost:8080/v1/tools/list_variants
```

---

## Installation Steps

### Step 1: Create the Local Mirror
UniContext works by mirroring a virtual hierarchy. Create the local directory structure:

```bash
mkdir -p unicontext/{resources,user,agent,session}
```

### Step 2: Configure the Agent Skill
The key to UniContext is the **Agent's Behavior**. You must provide the agent with the `UniContext Hierarchical Memory Skill`. This tells the agent to:
- Map `viking://` to `/unicontext/`.
- Search for `/abstract` before `/details`.

### Step 3: Seed Your First Memory
Create a sample memory to test the hierarchy:

1. **Create the abstract:** `echo "Summary of Project X" > unicontext/resources/project_x/abstract`
2. **Create the details:** `echo "Full documentation for Project X..." > unicontext/resources/project_x/details`
3. **Ingest via Tool:**
   ```bash
   # Use the search_multimodal tool to verify connectivity
   # Use ingest_asset to store the memory
   ```

---

## Manual Table Initialization (If needed)

If you are setting up a new environment, run the database initialization:

```bash
# From the project root
go run cmd/db-migrate/main.go
```

This ensures the `filesys` table exists with:
- `path TEXT PRIMARY KEY` (Handles `/unicontext/...` strings)
- `metadata JSONB` (Handles tiered context labels)
- `embedding vector(640)` (Optimized for Gemma 3)

---

## Verification

Run a query to ensure the system is correctly mapping the hierarchy:

```bash
# Semantic search for the new memory
# Verify the path returned matches the UniContext convention
```

The memory system is "enabled" once the agent starts using the hierarchical path prefixes for all search and ingestion operations.
