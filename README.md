# DocIntel Agents — Multi-Agent Document Intelligence over PDFs

Planner → Researcher → Analyst → Verifier agents (LangGraph) answering questions over real
research papers, with FAISS semantic retrieval, MCP tool server, citations, and a
verification loop that re-retrieves when answers are not grounded in evidence.

```
PDFs → pypdf → sentence-aware chunks → MiniLM embeddings → FAISS
Question → planner (sub-queries) → researcher (tools: search_documents) → analyst (cited answer)
        → verifier (per-claim support check) ──fail──► researcher (wider k, max 2 attempts)
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
make setup      # installs deps (torch + sentence-transformers are large)
make all        # downloads 9 arXiv papers, builds FAISS index, runs eval -> dashboard/data/results.json
make test       # fast unit tests (offline hashing embedder)
make serve      # API on :8000, dashboard at http://localhost:8000
make mcp        # MCP server over stdio (tools: search_documents, get_chunk, list_documents)
```
Optional: `export GEMINI_API_KEY=...` switches the planner/analyst to LLM mode. Without a
key the system runs in extractive mode (answers are sentences selected from evidence).

## Publish
1. `git init && git add . && git commit -m "init" && git push` — CI runs tests, the full eval
   (downloads papers, builds index, enforces quality gates) and a Docker build.
   Add `GEMINI_API_KEY` (free from Google AI Studio) as a repo secret to evaluate in LLM mode.
2. Commit `dashboard/data/results.json` from your local `make eval` so the dashboard has data.
3. **Vercel** (dashboard): import the repo; `vercel.json` already sets `outputDirectory=dashboard`.
4. **Backend** (optional live queries): Vercel cannot host torch + FAISS (250 MB function
   limit). Deploy the `Dockerfile` to Render / Fly.io / Hugging Face Spaces (Docker), then paste
   the URL into the dashboard's "Ask the live system" box.

## Layout
`docintel/` ingest, embeddings, index, agents, tools, mcp_server, api, evaluate · `tests/` ·
`evaluation/questions.json` · `dashboard/` · `.github/workflows/ci.yml`
