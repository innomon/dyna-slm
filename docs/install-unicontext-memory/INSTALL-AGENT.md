# UniContext Memory Plugin — Agent Install Guide

> For AI Agents (Gemini CLI, Claude Code, Cursor, etc.) to follow step-by-step.
> Principle: Use the existing Dyna-SLM tools to implement the UniContext mapping.

---

## Step 0: Choose Target Environment

Confirm the target `gomlx-pgvect-rag` workspace:

```bash
ls -d /home/innomon/AB2026Dev/dyna-slm
```

If multiple directories exist, ask the user which workspace to use.

## Step 1: Initialize the UniContext Root

The agent must maintain a local mirror of the virtual filesystem to facilitate ingestion.

```bash
mkdir -p /home/innomon/AB2026Dev/dyna-slm/unicontext/resources
mkdir -p /home/innomon/AB2026Dev/dyna-slm/unicontext/user/memories
```

---

## Step 2: Check Environment & Backend

Run each check. Every check must pass before continuing.

### 1. Go (Backend)
```bash
go version
```
- **Pass:** >= 1.23
- **Fail:** Ask user to install the latest Go.

### 2. PostgreSQL (Database)
```bash
psql -c "SELECT version();"
```
- **Pass:** Version output present.
- **Fail:** Ask user for their Postgres credentials and connection string.

### 3. Dyna-SLM Tools
```bash
./bin/dyna-slm version
```
- **Pass:** Version output present.
- **Fail:** Compile the tools using `go build -o bin/ ./cmd/...`.

---

## Step 3: Configure Mapping (Behavioral Only)

Unlike traditional plugins, UniContext is a **mapping convention**.

### 1. Activate the Skill
Load `docs/install-unicontext-memory/SKILL.md` into your system context. This tells you how to:
- Map `viking://` URIs to `/unicontext/` paths.
- Store L0 (Abstract) and L1 (Overview) layers separately.

### 2. Test the Connection
Verify that the `search_multimodal` tool can communicate with the backend.

```bash
./bin/dyna-slm search --query "test"
```

---

## Step 4: Verify the Tiered Hierarchy

Run a manual ingestion of a tiered resource to confirm the setup:

1. **Create the files:**
   ```bash
   echo "ABSTRACT: A hierarchical memory system." > unicontext/resources/viking_intro/abstract
   echo "DETAILS: OpenViking provides a virtual filesystem..." > unicontext/resources/viking_intro/details
   ```

2. **Ingest via Tool:**
   ```bash
   # Use the ingest_asset tool for both
   # Ensure the path in the database is /unicontext/resources/viking_intro/abstract
   ```

3. **Verify in Database:**
   ```sql
   SELECT path FROM filesys WHERE path LIKE '/unicontext/%';
   ```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `path not found` | Local mirror mismatch | Ensure your local file path matches the intended virtual path |
| `search failed` | API service down | Restart the `cmd/api` server |
| `embedding failed` | Model weights missing | Run `python3 download_weights.py` |

Tell the user: "UniContext memory is active. I will now organize all new memories into the /unicontext/ hierarchy and use tiered context for efficient retrieval."
