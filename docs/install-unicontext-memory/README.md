# UniContext Hierarchical Memory for Dyna-SLM

Use UniContext as the hierarchical, filesystem-based long-term memory mapping for the Dyna-SLM RAG system. This implementation follows the OpenViking paradigm by organizing context into a virtual filesystem structure within the existing PostgreSQL database.

## Table of Contents

- Overview
- Prerequisites
- Installation & Setup
- Tiered Context (L0/L1/L2)
- Usage Guidelines
- Verification
- Troubleshooting

---

## Overview

UniContext evolves the Dyna-SLM RAG system from a "flat" retrieval model to a "hierarchical" exploration model. It solves context fragmentation and token inefficiency by:
1. **Virtual Mapping:** Mapping `viking://` URIs to `/unicontext/` paths in the `filesys` table.
2. **Tiered Layers:** Providing Abstract (L0), Overview (L1), and Details (L2) for every resource.
3. **Scan-Before-Read:** Enabling agents to scan summaries before committing to large context windows.

---

## Prerequisites

| Component | Requirement | Purpose |
|-----------|-------------|---------|
| **Go** | >= 1.23 | Backend runtime |
| **PostgreSQL** | >= 15 + pgvector | Vector storage |
| **GoMLX** | XLA-accelerated | Embedding generation |
| **Dyna-SLM** | Installed & Configured | Base RAG system |

---

## Installation & Setup

Since UniContext is a design pattern applied to the existing `gomlx-pgvect-rag` codebase, "installation" involves preparing your environment to support the hierarchical structure.

### 1. Initialize the Filesystem Root
Create a local directory to host your tiered context files before ingestion:

```bash
mkdir -p /home/innomon/AB2026Dev/dyna-slm/unicontext/resources
mkdir -p /home/innomon/AB2026Dev/dyna-slm/unicontext/user/memories
```

### 2. Configure Your Agent
Ensure your agent is using the `UniContext Hierarchical Memory Skill` (`SKILL.md`). This skill provides the behavioral instructions needed to navigate the `/unicontext/` hierarchy.

---

## Tiered Context (L0/L1/L2)

To maximize efficiency, always store resources in three layers:

- **L0 (Abstract):** Short summary (~100 tokens). Path: `/unicontext/.../abstract`
- **L1 (Overview):** Detailed overview (~2k tokens). Path: `/unicontext/.../overview`
- **L2 (Details):** Full original content. Path: `/unicontext/.../details`

---

## Usage Guidelines

### Storing Memory
Use the `ingest_asset` tool to add layers to the hierarchy. Ensure the local file path mirrors the desired virtual path.

```bash
# Example: Ingesting an abstract for a project spec
ingest_asset --model dyna-gemma3-270m --path /path/to/unicontext/resources/project_a/spec.md/abstract
```

### Recalling Memory
1. **Semantic Search:** Query the system using `search_multimodal`.
2. **Filter by Path:** Look for results where `path` starts with `/unicontext/` and ends with `/abstract`.
3. **Drill Down:** If relevant, fetch the `/overview` or `/details` for that specific resource.

---

## Verification

### Database Check
Verify that your entries are correctly mapped in PostgreSQL:

```sql
SELECT path, metadata->>'layer' as layer FROM filesys WHERE path LIKE '/unicontext/%';
```

### Tool Verification
Run a test search to ensure the agent can "see" the hierarchy:

```bash
./bin/dyna-slm search --query "project a status"
# Should return /unicontext/resources/project_a/spec.md/abstract
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No results for `viking://` | Path mapping mismatch | Ensure all ingested paths start with `/unicontext/` |
| Token overflow | Loading L2 directly | Follow the L0 -> L1 -> L2 progression |
| Search is slow | Missing index | Verify `idx_filesys_metadata` (GIN) is active |
